## AI调用示例

```多模态大模型调用示例

/*这个模型可以识别图片，并根据输入的文字指令完成任务*/

from openai import OpenAI

client = OpenAI(

base_url='https://api-inference.modelscope.cn/v1',

api_key = "<MODELSCOPE_TOKEN>" # ModelScope Token

)

response = [client.chat](http://client.chat/).completions.create(

model='Qwen/Qwen3-VL-235B-A22B-Instruct', # ModelScope Model-Id, required

messages=[{

'role':

'user',

'content': [{

'type': 'text',

'text': '描述这幅图',

}, {

'type': 'image_url',

'image_url': {

'url':

'https://modelscope.oss-cn-beijing.aliyuncs.com/demo/images/audrey_hepburn.jpg',

},

}],

}],

stream=True

)

for chunk in response:

if chunk.choices:

print(chunk.choices[0].delta.content, end='', flush=True)

```