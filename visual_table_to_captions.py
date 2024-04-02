from openai import OpenAI
import base64
import json
import copy
from tqdm import tqdm

example = r"""
{
  "scene description": "Two children sitting on a couch indoors, likely in a living room, appearing to be engaged in a video game session. The time of day is not discernible from the image.",
  "objects": [
    {
      "object category": "Human",
      "attribute description": "Two children, sitting, focused expressions, casual clothing, barefoot",
      "knowledge description": "Children likely engaged in leisure activity, possibly playing video games which can improve hand-eye coordination and problem-solving skills"
    },
    {
      "object category": "Couch",
      "attribute description": "Large, leather material, dark color",
      "knowledge description": "Furniture designed for seating multiple people, commonly used in living areas for relaxation and socializing"
    },
    {
      "object category": "Video game controllers",
      "attribute description": "Held by each child, ergonomic design, associated with interactive entertainment",
      "knowledge description": "Input devices used to interact with video games, providing a means for users to control characters or elements in the game"
    },
    {
      "object category": "Striped pants",
      "attribute description": "Worn by one child, black and white stripes, fitted style",
      "knowledge description": "Clothing item, likely chosen for comfort or personal style, stripes can be a choice for aesthetic appeal"
    },
    {
      "object category": "Graphic T-shirt",
      "attribute description": "Worn by one child, printed design on the front, short sleeves",
      "knowledge description": "Casual upper garment, often used to express personal interests or tastes through graphic designs"
    },
    {
      "object category": "Floor",
      "attribute description": "Wooden texture, light color, horizontal planks",
      "knowledge description": "Part of home construction, provides a durable and stable surface for furniture and walking"
    }
  ]
}
"""

PROMPT_TEMPLATE = """
Create detailed captions describing the contents of the given image. 
Include the object types and colors, counting the objects, object actions, precise object locations, texts and doublechecking relative positions between objects.
Instead of describing the imaginary content, only describing the content one can determine confidently from the image. Please objectively describe what really exists, don't use aesthetic descriptions. 
Please carefully check the relative position between objects.
Do not describe the contents by itemizing them in list form.
Please do not include specific coordinate descriptions in the answer.
Some auxiliary information including the category of objects and the location of detection boxes are given. 

Here are some examples:

[bounding box]
control: [0.483, 0.482, 0.511, 0.531] control: [0.147, 0.483, 0.266, 0.705] control: [0.553, 0.429, 0.618, 0.462] pillow: [0.235, 0.259, 0.541, 0.519] pillow: [0.74, 0.35, 0.806, 0.496] pillow: [0.544, 0.146, 0.846, 0.244] pillow: [0.788, 0.371, 0.997, 0.562] remote_control: [0.148, 0.483, 0.265, 0.703] remote_control: [0.553, 0.427, 0.617, 0.46] 

[visual_table]
{example}


[resopnse]
The image depicts an indoor setting with two children sitting on a dark-colored sofa. Both children are looking directly at the camera with neutral expressions. The child on the left is wearing a short-sleeved top with a pattern and striped leggings, while the child on the right is wearing a T-shirt with the letters \"NY\" and solid-colored pants. They are both barefoot, and their feet are prominently positioned towards the camera. There are several pillows on the sofa: one is positioned behind the child on the left, another is partially visible behind the child on the right, and two additional pillows are placed on the right end of the sofa. There are two remote controls in the scene; one is held by the child on the left, and the other is held by the child on the right. The overall color palette of the image is monochromatic, suggesting that the photo is in black and white.

Now finish your tasks:

[tag]
{tag}

[bounding box]
{bbox}

[visual_table]
{visual_table}



[resopnse]
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
    data = [json.loads(item) for item in open("gpt4v result\\visual_table.jsonl", "r", encoding="utf-8")]

    output = []
    sample_dict = {}
    for sample in tqdm(data):
        sample_dict.clear()
        prompt = PROMPT_TEMPLATE.format(example=example, tag = sample['tag'], bbox=sample['bbox'], visual_table=sample['visual_table'])
        sample_dict['image_path'] = sample['image_path']
        sample_dict['captions'] = get_output(prompt, encode_image(image_path="annotations\\test_image\\" + sample["image_path"]))
        output.append(copy.deepcopy(sample_dict))

        with open("gpt4v result\\visual_table_to_captions.jsonl", "w", encoding="utf-8") as f:
            for sample in output:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')