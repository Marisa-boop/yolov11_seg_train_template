1. add data to datasets_origin
2. add template_mask_image to template_image
```
datasets_origin
- 1
  - images
  - masks
- 2
  - images
  - masks
template_image
- ?.jpg
...
```

3. `python ./utils/merge_data.py`
4. `python ./utils/masks2yoloForm.py`
5. `python ./utils/spilit_data.py`
6. `python ./utils/train.py`
7. if use coco:
- `python ./utils/coco2yoloSegForm.py --json json_file_path --output added_data/labels`
```
added_data
- images
- labels
```
- `python ./utils/updateDatasets.py`
add images and labels from added_data to merge_data

