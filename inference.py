import argparse
import torch
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info


MODEL_PATH = None
llm = None
processor = None
tokenizer = None


def initialize(model_path: str):
    global llm, processor, tokenizer, MODEL_PATH
    MODEL_PATH = model_path

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=81920,
        gpu_memory_utilization=0.9,
        limit_mm_per_prompt={"image": 10, "video": 10}
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        max_tokens=1024,
        stop_token_ids=[]
    )

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer

    return sampling_params


def run_inference(
    messages: list,
    sampling_params: SamplingParams,
    return_video_kwargs: bool = False
):
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    llm_inputs = []
    mm_type = messages[0]['content'][0]['type']

    if mm_type == 'image':
        sample_mm_data = {"image": image_inputs[0]} if image_inputs else {}
        sample_video_kw = {}
    elif mm_type == 'video':
        sample_mm_data = {"video": video_inputs[0]} if video_inputs else {}
        sample_video_kw = {key: val[0] for key, val in video_kwargs.items()}
    else:
        sample_mm_data = {}
        sample_video_kw = {}

    llm_inputs.append({
        "prompt": prompt,
        "multi_modal_data": sample_mm_data,
        "mm_processor_kwargs": sample_video_kw,
    })

    outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
    output_text = outputs[0].outputs[0].text

    return output_text


def infer_image(
    image_path: str,
    question: str,
    problem_type: str = "free-form",
    model_path: str = None
):
    if model_path is None:
        raise ValueError("model_path is required")

    sampling_params = initialize(model_path)

    QUESTION_TEMPLATE = (
        "{Question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
        "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
    )

    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "segmentation": " Please identify the referred object or region and provide the final result within the <answer> </answer> tags. The answer must include the literal token [SEG], for example <answer>the referred object [SEG]</answer>.",
    }

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE.get(problem_type, TYPE_TEMPLATE["free-form"])}
            ]
        }
    ]

    return run_inference(messages, sampling_params)


def infer_video(
    video_path: str,
    question: str,
    problem_type: str = "free-form",
    model_path: str = None,
    max_pixels: int = 200704,
    nframes: int = 32
):
    if model_path is None:
        raise ValueError("model_path is required")

    sampling_params = initialize(model_path)

    QUESTION_TEMPLATE = (
        "{Question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
        "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
    )

    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "segmentation": " Please identify the referred object or region and provide the final result within the <answer> </answer> tags. The answer must include the literal token [SEG], for example <answer>the referred object [SEG]</answer>.",
    }

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": max_pixels,
                    "nframes": nframes
                },
                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE.get(problem_type, TYPE_TEMPLATE["free-form"])}
            ]
        }
    ]

    return run_inference(messages, sampling_params)


def infer_text(
    question: str,
    problem_type: str = "free-form",
    model_path: str = None
):
    if model_path is None:
        raise ValueError("model_path is required")

    sampling_params = initialize(model_path)

    QUESTION_TEMPLATE = (
        "{Question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
        "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
    )

    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "segmentation": " Please identify the referred object or region and provide the final result within the <answer> </answer> tags. The answer must include the literal token [SEG], for example <answer>the referred object [SEG]</answer>.",
    }

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE.get(problem_type, TYPE_TEMPLATE["free-form"])}
            ]
        }
    ]

    return run_inference(messages, sampling_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sa2VA_Pro Inference")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model")
    parser.add_argument('--video_path', type=str, default=None, help="Path to input video")
    parser.add_argument('--image_path', type=str, default=None, help="Path to input image")
    parser.add_argument('--question', type=str, required=True, help="Question to ask")
    parser.add_argument('--problem_type', type=str, default="free-form",
                       choices=["multiple choice", "numerical", "OCR", "free-form", "regression"],
                       help="Type of problem")
    parser.add_argument('--max_pixels', type=int, default=200704, help="Max pixels for video frames")
    parser.add_argument('--nframes', type=int, default=32, help="Max number of video frames")

    args = parser.parse_args()

    if args.video_path:
        result = infer_video(
            video_path=args.video_path,
            question=args.question,
            problem_type=args.problem_type,
            model_path=args.model_path,
            max_pixels=args.max_pixels,
            nframes=args.nframes
        )
    elif args.image_path:
        result = infer_image(
            image_path=args.image_path,
            question=args.question,
            problem_type=args.problem_type,
            model_path=args.model_path
        )
    else:
        result = infer_text(
            question=args.question,
            problem_type=args.problem_type,
            model_path=args.model_path
        )

    print(result)
