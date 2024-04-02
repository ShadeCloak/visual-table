from openai import OpenAI
import base64
import json
import copy
from tqdm import tqdm

PROMPT_TEMPLATE = """
Create detailed captions describing the contents of the given image. 
Include the object types and colors, counting the objects, object actions, precise object locations, texts and doublechecking relative positions between objects.
Instead of describing the imaginary content, only describing the content one can determine confidently from the image. Please objectively describe what really exists, don't use aesthetic descriptions. 
Please carefully check the relative position between objects.
Do not describe the contents by itemizing them in list form.
Please do not include specific coordinate descriptions in the answer.
Some auxiliary information including the category of objects and the location of detection boxes are given. 

Here are some examples:
[Key words]
toothbrush.mirror.spectacles.camera

[bounding box]
toothbrush: [0.22, 0.61, 0.318, 0.749] mirror: [0.003, 0.003, 0.821, 0.995] spectacles: [0.709, 0.585, 0.74, 0.697] camera: [0.116, 0.43, 0.228, 0.542]

[resopnse]
The image shows a bathroom setting. On the wall, there is a large mirror covering most of the visible area. Reflected in the mirror is a person wearing a black T-shirt with some printed text or graphics. Near the top right corner of the mirror, a pair of spectacles can be seen resting on a hook. The man is holding a toothbrush in his left hand and appears to be brushing his teeth. The wall tiles have a decorative border pattern running along the top. The overall lighting in the scene seems somewhat dim or reddish in tone.

[Key words]
giraffe.flowers.

[bounding box]
giraffe: [0.731, 0.111, 1.0, 0.285] giraffe: [0.451, 0.373, 0.664, 0.974] 

[resopnse]
The image captures a scene with two giraffes and some flowers in the foreground. The giraffe on the right side of the image is partially visible with its head and neck extending from the top right corner to the center, showing its distinctive spots and ossicones. The other giraffe is located more centrally and is shown in full profile from its head down to its legs, with a clear view of its spotted coat. In the foreground, there is a cluster of slender-stemmed flowers with small purple blooms. The background includes a structure with an open doorway emitting light, and the setting appears to be an outdoor enclosure, possibly within a zoo environment.

Now finish your tasks:
[Key words]
{key_words}

[bounding box]
{bbox}

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
                        'role': 'system',
                        'content': 'You are a powerful image captioner.'
                    },
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
    data = [json.loads(item) for item in open("annotations\\gt.jsonl", "r", encoding="utf-8")]
    
    output = []
    for sample in tqdm(data):
        sample_dict = copy.deepcopy(sample)
        prompt = PROMPT_TEMPLATE.format(key_words=sample['tag'], bbox=sample['bbox'])
        response = get_output(prompt, encode_image(image_path="annotations\\test_image\\" + sample["image_path"].split('/')[-1]))
        sample_dict['captions'] = response
        output.append(copy.deepcopy(sample_dict))

        with open("gpt4v result\\gt_output.jsonl", "w", encoding="utf-8") as f:
            for sample in output:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')


