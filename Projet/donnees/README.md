---
license: cc-by-4.0
task_categories:
- image-classification
task_ids: []
pretty_name: Snacks
---

# Dataset Card for Snacks

## Dataset Summary

This is a dataset of 20 different types of snack foods that accompanies the book [Machine Learning by Tutorials](https://www.raywenderlich.com/books/machine-learning-by-tutorials/v2.0).

The images were taken from the [Google Open Images dataset](https://storage.googleapis.com/openimages/web/index.html), release 2017_11. 

## Dataset Structure

Number of images in the train/validation/test splits:

```nohighlight
train    4838
val      955
test     952
total    6745
```

Total images in each category:

```nohighlight
apple         350
banana        350
cake          349
candy         349
carrot        349
cookie        349
doughnut      350
grape         350
hot dog       350
ice cream     350
juice         350
muffin        348
orange        349
pineapple     340
popcorn       260
pretzel       204
salad         350
strawberry    348
waffle        350
watermelon    350
```

To save space in the download, the images were resized so that their smallest side is 256 pixels. All EXIF information was removed.

### Data Splits

Train, Test, Validation

## Licensing Information

Just like the images from Google Open Images, the snacks dataset is licensed under the terms of the Creative Commons license. 

The images are listed as having a [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) license. 

The annotations are licensed by Google Inc. under a [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. 

The **credits.csv** file contains the original URL, author information and license for each image.
