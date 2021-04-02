# Quantization

---

## image-selection
- File location: `${root_nestc}utils/imagenet_val_script/image_selection.py`
- `image-selection` generates a list file that contains paths of images for calibration.  
- Options you specify are the followings:
    - `--train-images-dir`
    - `--num-of-images`
    - `--file-name`
```bash
python image_selection.py --file-name=test.txt --num-of-images=1 --train-images=/home/jemin/hdd/imagenet/train_processed
```

## Imagenet sorting by labels (Optional)

`imagenet_sorting_by_labels.py` arranges images depending on labels.
- `val12_label.txt` contains labels of Imagenet

```bash
imagenet_sorting_by_labels.py <validation data dir> <validation labels file>
```
![sorting](docs/nestc/image_re_arrangement.png)

## imagenet_topk_accuracy_driver_py3

`imagenet_topk_accuracy_driver_py3.py` provides the following features:
- image preprocessing to fit 224x224x3 size of input tensor.
```bash
--validation-images-dir=/ssd/imagenet/train
--only-resize-and-save
--resize-mode=0

#output
Saving centered cropped input images: /ssd/imagenet/train/processed
0 / 1000
1 / 1000
2 / 1000
3 / 1000
4 / 1000
5 / 1000
6 / 1000
```
- Accuracy measurement with validation-set.
```bash
--verbose # log 저장 + resize 이미지 저장
--batch-size # 성능향상 없음
--validation-images-dir=/Users/jeminlee/fake_imagenet/
# 검증 데이터셋 위치. 레이블 오름차순으로 디텍토리 안에 정렬
--image-classifier-cmd="/Users/jeminlee/development/forked_glow/NEST_Compiler2/nest_compiler/cmake-build-release/bin/image-classifier -m=/Users/jeminlee/development/onnx_models/resnet-v1/resnet18v1.onnx -model-input-name=data -image-mode=0to1 -use-imagenet-normalization -backend=CPU -compute-softmax -topk=5 -”
# - streaming option을 반드시 줌.
# image에서 normalization을 하지 않았으므로 여기서 줘야함.
# image-mode는 0~1
# -topk=5를 해서 top5 error까지 계산 하도록 함.
```

To easily use `imagenet_topk_accuracy_driver_py3.py` and handle outputs, the following shell scripts are provided.
### Calibration
```bash
# resnet 18
cd /home/jemin/development/nest_compiler/utils/profile
./profile.sh
```

### Parameter Search

```bash
cd /home/jemin/development/nest_compiler/utils/imagenet_val_script

# 1,000, Max, non-fusion
./resnet18.sh 3

# 10,000, KL, fusion
./resnet18.sh 4

# 10,000, KL, Fusion
./resnet18.sh 7
```

### Show explored results

```bash``
# All search results of resnet18 
ls | grep resnet18

# 3
ls | grep -E "nonfusion.*resnet18.*_1000_.*none" | xargs cat
# 4
ls | grep -E "nonfusion.*resnet18.*_10000_.*KL" | xargs cat
# 7
ls | grep -E "vtainterpreter-fusion.*resnet18.*_10000_.*KL" | xargs cat

# Print all results
cd /home/jemin/development/nest_compiler/utils/imagenet_val_script/quant_result
ls | grep resnet18 | xargs more
```
