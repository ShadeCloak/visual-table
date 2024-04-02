from openai import OpenAI
import base64
import json
import copy
from tqdm import tqdm

PROMPT_TEMPLATE = """
You are an AI visual assistant that can analyze a single image.

Given an image, you need to perform the task of scene description.And then, you need to identify each object in the image.
For each object, you need to perform 3 tasks: object category recognition, attribute description generation, and knowledge description generation.
some auxiliary information as fallow:
[tag]
{tag}
[bounding box]
{bbox}

Scene description:1. Based on the given image, please provide a short and concise description for the scene in the image, such as the location, the time of the day (e.g.,morning, evening), the event, and so on.
Object category recognition:1. Based on the given image, please recognize the category for each object in the scene.2. Please cover as many objects as possible. 
The objects should cover not only the salient objects, but also the other objects such as the small objects, theobjects in the background, the objects that are partially occluded, and so on.
Attribute description generation:1. Based on the given image, please generate the visual attributes for each object.2. Visual attributes characterize the objects in images. They can be OCR characters on the object, spatial relations to surrounding objects, action relationsto surrounding objects, relative size compared to surrounding objects, color, geometry shape, material, texture pattern, scene environment, motion ordynamics of objects, and so on.3. Specially, if possible, the visual attributes could be the emotions (e.g., surprised, angry), age (e.g., young, elderly), and so on.Knowledge description generation:1. Based on the given image, please describe the knowledge for each object.2. The knowledge includes object affordance, commonsense knowledge, background knowledge, and so on.3. Object affordance is defined as the functions supported by the objects. For example, what the objects can be used for? Note that the affordance might bealtered case by case, due to deformed shape, unreliable materials, and so on.4. Commonsense knowledge is defined as basic understandings and assumptions about daily life, human behavior, and the natural world.It also includes understanding social norms, basic cause-and-effect relationships, and simple reasoning about daily situations.5. Background knowledge is defined as the knowledge of named entities, such as celebrities, ceremonies, festivals, and so on.

Output format:The output content should follow the following JSON format.{"scene description": "", "objects": [{"object category": "", "attribute description": "", "knowledge description": ""}, ......, {"object category":"", "attribute description": "", "knowledge description": ""}]}.
Directly output the JSON without any other content. The output MUST follow JSON format.
some auxiliary information as fallow…

"""
PROMPT_TEMPLATE2 = """
some auxiliary information as fallow:
[tag]
{tag}
[bounding box]
{bbox}
"""
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_output(prompt: str, base64_image: str, max_tokens: int=1024):
    
    while True:
        try:
            client = OpenAI(api_key='<KEY>')
            client.api_key = 'sk-31S7XYPRdVQt2OkrF4415d92E269416a8974CbF0BaC4A268'
            client.base_url = 'https://api.gptplus5.com/v1'
            response = client.chat.completions.create(
                model='gpt-4-vision-preview',
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {   
                                "type": "text", 
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                            }
                        ],
                    }
                ],
                temperature=0.2,  
                max_tokens=max_tokens,
            )
            break
        # except openai.error.RateLimitError:
        #     pass
        except Exception as e:
            print(e)

    return response.choices[0].message.content

if __name__ == '__main__':
    data = [json.loads(item) for item in open("annotations\\bbox_result_ram.jsonl", "r", encoding="utf-8")]

    output = []
    sample_dict = {}
    for sample in tqdm(data):
        sample_dict.clear()
        sample_dict['image_path'] = sample['image_path']
        sample_dict['tag'] = sample['tag']
        sample_dict['bbox'] = sample['bbox']
        prompt = PROMPT_TEMPLATE + PROMPT_TEMPLATE2.format(tag=sample_dict['tag'], bbox=sample_dict['bbox'])
        sample_dict['visual_table'] = get_output(prompt, encode_image(image_path="annotations\\test_image\\" + sample["image_path"].split('/')[-1]))
        output.append(copy.deepcopy(sample_dict))

        with open("gpt4v result\\visual_table.jsonl", "w", encoding="utf-8") as f:
            for sample in output:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')