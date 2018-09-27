#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <map>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/annotated_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/sampler.hpp"

namespace caffe {

template <typename Dtype>
AnnotatedDataLayer<Dtype>::AnnotatedDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param) {
}

template <typename Dtype>
AnnotatedDataLayer<Dtype>::~AnnotatedDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void AnnotatedDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param();

// 读取所有数据增强采样参数
  for (int i = 0; i < anno_data_param.batch_sampler_size(); ++i) {
    batch_samplers_.push_back(anno_data_param.batch_sampler(i));
  }
  label_map_file_ = anno_data_param.label_map_file();

  // Make sure dimension is consistent within batch.
  const TransformationParameter& transform_param =
    this->layer_param_.transform_param();

  if (transform_param.has_resize_param()) {
    if (transform_param.resize_param().resize_mode() ==
        ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
      CHECK_EQ(batch_size, 1)
        << "Only support batch size of 1 for FIT_SMALL_SIZE.";
    }
  }


// 读取一个数据，并读取数据的shape,初始化top的shape和prefetch的shape(比如数据大小为300x300)
// AnnotatedDatum包含了数据和标注(标注包含了label和bounding box)

  // Read a data point, and use it to initialize the top blob.
  AnnotatedDatum& anno_datum = *(reader_.full().peek());

  // Use data_transformer to infer the expected blob shape from anno_datum.
  vector<int> top_shape =
      this->data_transformer_->InferBlobShape(anno_datum.datum());
  this->transformed_data_.Reshape(top_shape);

  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);

// 预读线程中的图像数据
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

// label
  if (this->output_labels_) {
// 生成数据的时候是有类型的 anno_datum.set_type(AnnotatedDatum_AnnotationType_BBOX);
    has_anno_type_ = anno_datum.has_type() || anno_data_param.has_anno_type();
    vector<int> label_shape(4, 1);
    if (has_anno_type_) {
      anno_type_ = anno_datum.type();
      if (anno_data_param.has_anno_type()) {
        // If anno_type is provided in AnnotatedDataParameter, replace
        // the type stored in each individual AnnotatedDatum.
        LOG(WARNING) << "type stored in AnnotatedDatum is shadowed.";
        anno_type_ = anno_data_param.anno_type();
      }

      // Infer the label shape from anno_datum.AnnotationGroup().
      int num_bboxes = 0;
      if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
        // Since the number of bboxes can be different for each image,
        // we store the bbox information in a specific format. In specific:
        // All bboxes are stored in one spatial plane (num and channels are 1)
        // And each row contains one and only one box in the following format:
        // [item_id, group_label, instance_id, xmin, ymin, xmax, ymax, diff]
        // Note: Refer to caffe.proto for details about group_label and
        // instance_id.
        for (int g = 0; g < anno_datum.annotation_group_size(); ++g) {
          num_bboxes += anno_datum.annotation_group(g).annotation_size();
        }
        label_shape[0] = 1;
        label_shape[1] = 1;
        // BasePrefetchingDataLayer<Dtype>::LayerSetUp() requires to call
        // cpu_data and gpu_data for consistent prefetch thread. Thus we make
        // sure there is at least one bbox.
        label_shape[2] = std::max(num_bboxes, 1);
        label_shape[3] = 8;
      } else {
        LOG(FATAL) << "Unknown annotation type.";
      }
    } else {
      label_shape[0] = batch_size;
    }
    top[1]->Reshape(label_shape);

    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void AnnotatedDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first anno_datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param();
  const TransformationParameter& transform_param =
    this->layer_param_.transform_param();

// 初始化transformed_data_和 batch->data_的大小
  AnnotatedDatum& anno_datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from anno_datum.
  vector<int> top_shape =
      this->data_transformer_->InferBlobShape(anno_datum.datum());
  this->transformed_data_.Reshape(top_shape); // transformed_data_存储一幅图像，对于SSD300,transformed_data_大小为:[1,3,300,300]

  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);// batch->data_存储batchsize个图像,对于SSD300，batch->data_大小为[batchsize,3,300,300]

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_ && !has_anno_type_) {
    top_label = batch->label_.mutable_cpu_data();
  }

  // Store transformed annotation.
  map<int, vector<AnnotationGroup> > all_anno; // batchsize中每一幅图像以及对应的标注
  int num_bboxes = 0;

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();

// 获取一幅图像，并做相应的预处理(比如加入扰动)
    // get a anno_datum
    AnnotatedDatum& anno_datum = *(reader_.full().pop("Waiting for data"));
    read_time += timer.MicroSeconds();
    timer.Start();
    AnnotatedDatum distort_datum;
    AnnotatedDatum* expand_datum = NULL;


    if (transform_param.has_distort_param()) {
      distort_datum.CopyFrom(anno_datum);
      this->data_transformer_->DistortImage(anno_datum.datum(),
                                            distort_datum.mutable_datum());
      if (transform_param.has_expand_param()) {
        expand_datum = new AnnotatedDatum();
        this->data_transformer_->ExpandImage(distort_datum, expand_datum);
      } else {
        expand_datum = &distort_datum;
      }
    } else {
      if (transform_param.has_expand_param()) {
        expand_datum = new AnnotatedDatum();
        this->data_transformer_->ExpandImage(anno_datum, expand_datum);
      } else {
        expand_datum = &anno_datum;
      }
    }
    AnnotatedDatum* sampled_datum = NULL;
    bool has_sampled = false;


/* 1. 首先进行数据增强(对应论文2.2 Training部分的Data augmentation)
* 对于batchsize中的每一幅图像，为每个采样器(batch_sampler)生成max_sample个boundingbox(候选框)
* 每个采样器生成的boundingbox与目标的IOU=0.1,0.3,0.5,0.7,0.9，这个与论文的描述是一致的
* 示例：
  batch_sampler
  {
    sampler
    {
      min_scale: 0.3
      max_scale: 1.0
      min_aspect_ratio: 0.5
      max_aspect_ratio: 2.0
    }
    sample_constraint
    {
      min_jaccard_overlap: 0.7
    }
    max_sample: 1
    max_trials: 50
  }
*  对于该采样器，随机生成的满足条件的boundingbox与图像中任一目标的IOU>0.7
*  注意：
*    1). 生成的boundingbox坐标是归一化的坐标，这样不受resize的影响，目标检测的回归都是采用的这种形式(比如MTCNN)
*    2). 随机生成boundingbox的时候，根据每个batch_sampler的参数：尺度，宽高比，每个采样器最多尝试max_trials次
*
*/
    if (batch_samplers_.size() > 0) {

      // Generate sampled bboxes from expand_datum.
      vector<NormalizedBBox> sampled_bboxes;
      GenerateBatchSamples(*expand_datum, batch_samplers_, &sampled_bboxes);

/*2. 从生成的所有bounding box中随机挑选一个bounding box
* 裁剪出该bounding box对应的图像(大小就是sampled_bboxes[rand_idx]在原图中的大小)并计算该bounding box中所有目标的坐标以及类别
* 注意：
*    1). bounding box中目标的坐标=(原图中ground truth的坐标-该bounding box的坐标)/(bounding box的边长)
*    2). 这里groundtruth与boundingbox的坐标都相对于原图,在mtcnn中也是采用了该计算方式
*
*/
      if (sampled_bboxes.size() > 0) {
        // Randomly pick a sampled bbox and crop the expand_datum.
        int rand_idx = caffe_rng_rand() % sampled_bboxes.size();
        sampled_datum = new AnnotatedDatum();
        this->data_transformer_->CropImage(*expand_datum,
                                           sampled_bboxes[rand_idx],
                                           sampled_datum);
        has_sampled = true;
      } else {
        sampled_datum = expand_datum;
      }
    } else {
      sampled_datum = expand_datum;
    }
    CHECK(sampled_datum != NULL);
    timer.Start();
    vector<int> shape =
        this->data_transformer_->InferBlobShape(sampled_datum->datum());
    if (transform_param.has_resize_param()) {
      if (transform_param.resize_param().resize_mode() ==
          ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
        this->transformed_data_.Reshape(shape);
        batch->data_.Reshape(shape);
        top_data = batch->data_.mutable_cpu_data();
      } else {
        CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
              shape.begin() + 1));
      }
    } else {
      CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
            shape.begin() + 1));
    }

    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    vector<AnnotationGroup> transformed_anno_vec;
    if (this->output_labels_) {
      if (has_anno_type_) {
        // Make sure all data have same annotation type.
        CHECK(sampled_datum->has_type()) << "Some datum misses AnnotationType.";
        if (anno_data_param.has_anno_type()) {
          sampled_datum->set_type(anno_type_);
        } else {
          CHECK_EQ(anno_type_, sampled_datum->type()) <<
              "Different AnnotationType.";
        }


        // Transform datum and annotation_group at the same time
        transformed_anno_vec.clear();

/* 3. 将crop出来的AnnotatedDatum转换为数据部分和标注部分
*  数据部分会resize到数据层设置的大小(比如300x300)并保存到top[0]中
*  标注是所有目标在图像中的坐标
*
* 注意：
*  1. 这里的图像并不一定是原始crop的图像，如果transform_param有crop_size这个参数，原来crop出来的图像会再次crop的
*  2. 由于这里对crop出来的图像进行了一次resize,所以如果生成lmdb的时候，进行resize会导致数据层对原图进行两次resize，
*     这样有可能会影响到目标的宽高比，所以在SFD(Single Shot Scale-invariant Face Detector)中，对此处做了一点改进，即在第一步
*     生成boundingbox的时候，保证每个boundingbox都是正方形，这样resize到300x300的时候就不会改变目标的宽高比
*/

        this->data_transformer_->Transform(*sampled_datum,
                                           &(this->transformed_data_),
                                           &transformed_anno_vec);
        if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
          // Count the number of bboxes.
          for (int g = 0; g < transformed_anno_vec.size(); ++g) {
            num_bboxes += transformed_anno_vec[g].annotation_size();
          }
        } else {
          LOG(FATAL) << "Unknown annotation type.";
        }
        all_anno[item_id] = transformed_anno_vec;
      } else {
        this->data_transformer_->Transform(sampled_datum->datum(),
                                           &(this->transformed_data_));
        // Otherwise, store the label from datum.
        CHECK(sampled_datum->datum().has_label()) << "Cannot find any label.";
        top_label[item_id] = sampled_datum->datum().label();
      }
    } else {
      this->data_transformer_->Transform(sampled_datum->datum(),
                                         &(this->transformed_data_));
    }
    // clear memory
    if (has_sampled) {
      delete sampled_datum;
    }
    if (transform_param.has_expand_param()) {
      delete expand_datum;
    }
    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<AnnotatedDatum*>(&anno_datum));
  }



  // Store "rich" annotation if needed.
  if (this->output_labels_ && has_anno_type_) {
    vector<int> label_shape(4);
    if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
      label_shape[0] = 1;
      label_shape[1] = 1;
      label_shape[3] = 8;
      if (num_bboxes == 0) {
        // Store all -1 in the label.
        label_shape[2] = 1;
        batch->label_.Reshape(label_shape);
        caffe_set<Dtype>(8, -1, batch->label_.mutable_cpu_data());
      } else {
        // Reshape the label and store the annotation.
        label_shape[2] = num_bboxes;
        batch->label_.Reshape(label_shape);
        top_label = batch->label_.mutable_cpu_data();
        int idx = 0;
        for (int item_id = 0; item_id < batch_size; ++item_id) {
          const vector<AnnotationGroup>& anno_vec = all_anno[item_id];
          for (int g = 0; g < anno_vec.size(); ++g) {
            const AnnotationGroup& anno_group = anno_vec[g];
            for (int a = 0; a < anno_group.annotation_size(); ++a) {
              const Annotation& anno = anno_group.annotation(a);
              const NormalizedBBox& bbox = anno.bbox();
              top_label[idx++] = item_id;
              top_label[idx++] = anno_group.group_label();
              top_label[idx++] = anno.instance_id();
              top_label[idx++] = bbox.xmin();
              top_label[idx++] = bbox.ymin();
              top_label[idx++] = bbox.xmax();
              top_label[idx++] = bbox.ymax();
              top_label[idx++] = bbox.difficult();
            }
          }
        }
      }
    } else {
      LOG(FATAL) << "Unknown annotation type.";
    }
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(AnnotatedDataLayer);
REGISTER_LAYER_CLASS(AnnotatedData);

}  // namespace caffe
