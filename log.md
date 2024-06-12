```
# 尝试一些prompt看看效果

key:
base_url: 
在用4v生成文本描述的时候的prompt
"text": f"You are a powerful image captioner. Create detailed captions describing the contents of the given image. Include the object types and colors, counting the objects, object actions, precise object locations, texts and doublechecking relative positions between objects. Instead of describing the imaginary content, only describing the content one can determine confidently from the image. Do not describe the contents by itemizing them in list form. Please carefully check the relative position between objects. Please objectively describe what really exists, don't use aesthetic descriptions. Please do not include specific coordinate descriptions in the answer.\n Some auxiliary information including the category of objects and the location of detection boxes are as follows. {bbox_text}. These coordinates are in the form of bounding boxes, represented as [x1, y1, x2, y2] with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y."

你是一个强大的图像文本描述生成器。请根据给定的图像创建详细的文本描述，描述图像的内容。包括物体类型和颜色，计算物体数量，物体动作，精确的物体位置，图中的文本以及检查物体之间的相对位置。不要描述虚构的内容，只描述可以从图像中确定的内容。请客观地描述真实存在的东西，不要带有美学色彩。请不要在答案中包含具体的坐标形式。不要以列表形式描述内容。不要生成与图像内容无关的话。一些辅助信息包括物体的类别和检测框的位置，如下所示：{bbox_text}。这些坐标以边界框的形式表示为[x1, y1, x2, y2],其中浮点数范围从0到1。这些值对应于左上角的x坐标、左上角的y坐标、右下角的x坐标和右下角的y坐标。
```



目前prompt的效果：对于一些图像中的小细节仍有一点偏差（幻觉问题），特别是下面几个场景：1. 镜像问题，gpt4v会认为是两个人，类似这样的问题。2. 偶尔图片上物体之间重叠，但在空间上是相互错开的，但gpt4v认为物体相互朝向或者其他情况。3. 偶尔会出现相对位置的偏差，比如左侧和右侧，有时候分不清。4. 如果是单个物体，一般描述效果挺好，物体多会出现一些细节错误。



```
原始响应：In the image, we see a kitchen scene with a large white bird, possibly a heron or egret, standing on a patterned floor in front of an open drawer. To the left of the bird, there is a refrigerator that takes up almost half of the image's left side. The refrigerator is covered with various magnets and a single bird-shaped one is notably separated on the top left of the fridge. Directly in front of the bird, a stove with black burners is visible and it extends across the right side of the image, with a frying pan on one of the burners. Above the stove, there is a toaster oven placed on the counter. The counter is also the location of two salt shakers, one slightly to the left and behind the other. In the background, there is a wall socket visible to the right of the bird and slightly above the counter level. There are three drawers in the image, with the open drawer directly under the bird, and the other two are closed and are located to the right of the kitchen scene, with the second drawer being visible and the third is partially cut off by the edge of the image.


GPT-3.5提取实体得到：
bird.floor.drawer.refrigerator.magnet.stove.burner.frying pan.toaster oven.counter.salt shaker.wall socket.drawers


GPT-3.5产生问题：
What color is the bird?&bird
What is on top of the refrigerator?&refrigerator
What is on the counter?&counter
How many salt shakers are there?&salt shaker
What is on the burner?&burner
Is the drawer closed?&drawer
Is the toaster oven on the counter?&toaster oven.counter
Is the stove black?&stove


调用工具或其他模型来回答上面产生的问题：
The bird in the image is white.
On top of the refrigerator, there is no information available in the given image analysis.
On the counter, there are a stove and other appliances.
Based on the image analysis, there are no salt shakers visible in the image.
There is a flame visible on the burner.
yes, the drawer is closed.
Yes, the toaster oven is on the counter.
Yes, the stove in the image is black in color.

...
...
```



