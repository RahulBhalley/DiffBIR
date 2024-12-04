"""Image captioning utilities.

This module provides classes for generating text captions from images using
different models like LLaVA and RAM.
"""

from typing import overload, Literal
import re

from PIL import Image
import torch

try:
    from llava.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
        IMAGE_PLACEHOLDER,
    )
    from llava.conversation import conv_templates
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import (
        process_images,
        tokenizer_image_token,
        get_model_name_from_path,
    )

    LLAVA_AVAILABLE = True
except Exception as e:
    print(f"failed to import llava, error: {e}")
    LLAVA_AVAILABLE = False


try:
    from ram.models import ram_plus
    from ram import inference_ram as inference
    from ram import get_transform

    RAM_AVAILABLE = True
except Exception as e:
    print(f"failed to import ram, error: {e}")
    RAM_AVAILABLE = False


class Captioner:
    """Base class for image captioning models.

    This class defines the interface that all captioning models must implement.
    Subclasses should override the __call__ method to provide specific captioning
    functionality.

    Args:
        device (torch.device): The device to run the model on (CPU/GPU).
    """

    def __init__(self, device: torch.device) -> "Captioner":
        self.device = device

    @overload
    def __call__(self, image: Image.Image) -> str: ...


class EmptyCaptioner(Captioner):
    """A captioner that returns an empty string.

    This captioner is useful as a fallback when other captioning models are not
    available or as a placeholder during testing.
    """

    def __call__(self, image: Image.Image) -> str:
        """Return an empty caption regardless of input.

        Args:
            image (PIL.Image.Image): The input image to caption.

        Returns:
            str: An empty string.
        """
        return ""


class LLaVACaptioner(Captioner):
    """Image captioner using the LLaVA (Large Language and Vision Assistant) model.

    This captioner uses LLaVA to generate natural language descriptions of images.
    It supports different quantization levels (4-bit, 8-bit, 16-bit) for memory efficiency.

    Args:
        device (torch.device): The device to run the model on (CPU/GPU).
        llava_bit (Literal["16", "8", "4"]): Bit precision for model quantization.
    """

    def __init__(
        self, device: torch.device, llava_bit: Literal["16", "8", "4"]
    ) -> "LLaVACaptioner":
        super().__init__(device)
        if llava_bit == "16":
            load_4bit, load_8bit = False, False
        elif llava_bit == "8":
            load_4bit, load_8bit = False, True
        else:
            load_4bit, load_8bit = True, False

        model_path = "liuhaotian/llava-v1.5-7b"
        model_name = get_model_name_from_path(model_path)
        device_map = {"": device}
        self.tokenizer, self.model, self.image_processor, context_len = (
            load_pretrained_model(
                model_path,
                None,
                model_name,
                device=device,
                device_map=device_map,
                load_4bit=load_4bit,
                load_8bit=load_8bit,
            )
        )
        self.model.eval()

        qs = "Please give me a very short description of this image."
        image_token_se = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        self.prompt = conv.get_prompt()

        self.temperature = 0
        self.top_p = None
        self.num_beams = 1
        self.max_new_tokens = 512

    @torch.no_grad()
    def __call__(self, image: Image.Image) -> str:
        """Generate a natural language caption for the input image.

        Args:
            image (PIL.Image.Image): The input image to caption.

        Returns:
            str: A natural language description of the image.
        """
        images = [image]
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images, self.image_processor, self.model.config
        ).to(self.device, dtype=torch.float16)
        input_ids = (
            tokenizer_image_token(
                self.prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0)
            # .repeat(batch_size, 1)
            .to(self.device)
        )
        output_ids = self.model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if self.temperature > 0 else False,
            temperature=self.temperature,
            top_p=self.top_p,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
        )
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        res = [s.strip() for s in outputs]
        return res[0]


class RAMCaptioner(Captioner):
    """Image captioner using the RAM (Recognize Anything Model) Plus model.

    This captioner uses the RAM Plus model to generate tag-based descriptions of images.
    It outputs a comma-separated list of objects and concepts detected in the image.

    Args:
        device (torch.device): The device to run the model on (CPU/GPU).
    """

    def __init__(self, device: torch.device) -> Captioner:
        super().__init__(device)
        image_size = 384
        transform = get_transform(image_size=image_size)
        pretrained = "https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth"
        model = ram_plus(pretrained=pretrained, image_size=image_size, vit="swin_l")
        model.eval()
        model = model.to(device)

        self.transform = transform
        self.model = model

    def __call__(self, image: Image.Image) -> str:
        """Generate a tag-based caption for the input image.

        Args:
            image (PIL.Image.Image): The input image to caption.

        Returns:
            str: A comma-separated list of objects and concepts detected in the image.
        """
        image = self.transform(image).unsqueeze(0).to(self.device)
        res = inference(image, self.model)
        # res[0]: armchair | blanket | lamp | ...
        # res[1]: 扶手椅  | 毯子/覆盖层 | 灯  | ...
        return res[0].replace(" | ", ", ")
