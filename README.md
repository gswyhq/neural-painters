
AI印象派画作

运行：
`python3 intrinsic_style_transfer.py F561f22668fee4.jpg`
运行完后，会在当前目录生成一个名为`tmp.mp4`的输出文件

或者在docker中运行：
`docker run --rm -it -v $PWD:/tmp gswyhq/neural-painters python3 intrinsic_style_transfer.py /tmp/234513223.jpg`

原代码是在GPU上运行的，改成在CPU上运行，耗时超过48h。

