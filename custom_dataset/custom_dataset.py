"""custom_dataset dataset."""

import tensorflow_datasets as tfds
from tensorflow_datasets.core.features import BBoxFeature
import xmltodict
from PIL import Image
import numpy as np
import tensorflow as tf

# TODO(custom_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(custom_dataset): BibTeX citation
_CITATION = """
"""
LABEL=['Choi Woo-shik','Kim Da-mi','Kim Seong-cheol','Kim Tae-ri','Nam Joo-hyuk','Yoo Jae-suk']


class CustomDataset(tfds.core.GeneratorBasedBuilder):
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
   data.zip files should be located at /root/tensorflow_dataset/downloads/manual
   """

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(custom_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(None, None, 3)),
            'grid': tfds.features.Tensor(shape=(13,13,5,11), dtype=tf.float32),
            'objects': tfds.features.Sequence({'label': tfds.features.ClassLabel(names=LABEL)})
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'grid', 'objects'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(custom_dataset): Downloads the data and defines the splits

    custom_path = dl_manager.manual_dir / 'data.zip'
    custom_path = dl_manager.extract(custom_path)

    # TODO(custom_dataset): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(img_path=custom_path / 'train_images',
                                        xml_path=custom_path / 'train_xmls'),
        'test': self._generate_examples(img_path=custom_path / 'test_images',
                                        xml_path=custom_path / 'test_xmls')
    }

  def _generate_examples(self, img_path, xml_path):
    """Yields examples."""
    # TODO(custom_dataset): Yields (key, example) tuples from the dataset
    for i, (img, xml) in enumerate(zip(img_path.glob('*.jpg'), xml_path.glob('*.xml'))):
      yield i,{
        'image': img,
        'grid': self._get_grid(xml),
        'objects': self._get_label(xml)
      }
  
  def _get_grid(self, xml):
    anchors = [[1.4794683, 1.35702793], [1.58223277, 2.75638217], [2.88642237, 2.30851217], [2.51967881, 4.26763735], [4.82422153, 4.74087758]]
    anchors = np.reshape(anchors, (5,2)).astype(np.float32)
    f=open(xml)
    xml_file=xmltodict.parse(f.read())
    grid=np.zeros((13,13,5,11), np.float32)
    height, width = xml_file['annotation']['size']['height'], xml_file['annotation']['size']['width']
    objs = np.reshape(list([xml_file['annotation']['object']]),(-1,1))
    for obj in objs:
      obj=obj[0]
      if type(obj)==type(dict()):
        x1=obj['bndbox']['xmin']
        y1=obj['bndbox']['ymin']
        x2=obj['bndbox']['xmax']
        y2=obj['bndbox']['ymax']
        y1, y2 = float(y1)/float(height), float(y2)/float(height)
        x1, x2 = float(x1)/float(width), float(x2)/float(width)
        by, bx = (y1+y2)/2*13, (x1+x2)/2*13
        bh, bw = (y2-y1)*13, (x2-x1)*13
  
        intersection = np.maximum(np.minimum(bw, anchors[:,0]) * np.minimum(bh, anchors[:,1]), 0.0)
        union = anchors[:, 0] * anchors[:, 1] + bw*bh - intersection
        best_anchor = np.argmax(intersection/union, -1)

        cy, cx = int(by), int(bx)
        grid[cy,cx,best_anchor,0:4] = [bx, by, bw, bh]
        grid[cy,cx,best_anchor,4] = 1.0
        grid[cy,cx,best_anchor,5+LABEL.index(obj['name'])] = 1.0
    f.close()
    return grid
  def _get_label(self, xml):
    dic=dict()
    f=open(xml)
    xml_file=xmltodict.parse(f.read())
    label=[]
    for obj in xml_file['annotation']['object']:
      if type(obj)==type(dict()):
        label.append(obj['name'])
      else:
        if obj=='name':
          label.append(xml_file['annotation']['object'][obj])
    f.close()
    dic['label']=label
    return dic