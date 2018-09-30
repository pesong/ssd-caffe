#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <map>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/annotated_data_with_seg_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/sampler.hpp"

namespace caffe {

template <typename Dtype>
AnnotatedDataWithSegLayer<Dtype>::AnnotatedDataWithSegLayer(const LayerParameter& param)
  : BasePrefetchingDataWithSegLayer<Dtype>(param),
    reader_(param) {
}

template <typename Dtype>
AnnotatedDataWithSegLayer<Dtype>::~AnnotatedDataWithSegLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void AnnotatedDataWithSegLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    LOG(INFO) << "--------datalayer setup";
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
  vector<int> top_shape = this->data_transformer_->InferBlobShape(anno_datum.datum());
  this->transformed_data_.Reshape(top_shape);

  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);



    //added by pesong:  read label image
    // Use data_transformer to infer the expected blob shape from anno_datum.
  vector<int> top_shape_label = this->data_transformer_->InferBlobShape(anno_datum.datum_label());
  this->transformed_label_img_.Reshape(top_shape_label);

    // Reshape top[1] and prefetch_data according to the batch_size.
  top_shape_label[0] = batch_size;
  top[2]->Reshape(top_shape_label);



    // 预读线程中的图像数据
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }

    // 预读线程中的seg label image  added by pesong
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_img_.Reshape(top_shape_label);
    }

  LOG(INFO) << "----[top0]output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

LOG(INFO) << "[top2]output lable img size: " << top[2]->num() << ","
<< top[2]->channels() << "," << top[2]->height() << ","
<< top[2]->width();

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
        // BasePrefetchingDataWithSegLayer<Dtype>::LayerSetUp() requires to call
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
void AnnotatedDataWithSegLayer<Dtype>::load_batch(Batch<Dtype>* batch) {

  LOG(INFO) << "----start load_batch-------- ";
  LOG(INFO) << "batch->label_img_.count "<<batch->label_img_.count();

  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(batch->label_img_.count());

  CHECK(this->transformed_data_.count());
  CHECK(this->transformed_label_img_.count());



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
  vector<int> top_shape = this->data_transformer_->InferBlobShape(anno_datum.datum());
  this->transformed_data_.Reshape(top_shape); // transformed_data_存储一幅图像，对于SSD300,transformed_data_大小为:[1,3,300,300]

  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);// batch->data_存储batchsize个图像,对于SSD300，batch->data_大小为[batchsize,3,300,300]


    //added by pesong
  vector<int> top_shape_label = this->data_transformer_->InferBlobShape(anno_datum.datum_label());
  this->transformed_label_img_.Reshape(top_shape_label); // transformed_label_img_，对于SSD300,transformed_data_大小为:[1,3,300,300]

    // Reshape batch according to the batch_size.
  top_shape_label[0] = batch_size;
  batch->label_img_.Reshape(top_shape_label);
//  LOG(INFO) << "-----top_shape_label:---------" <<top_shape_label[0]<<top_shape_label[1]<<top_shape_label[2]<<top_shape_label[3];


  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label_img = batch->label_img_.mutable_cpu_data();

  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_ && !has_anno_type_) {
    top_label = batch->label_.mutable_cpu_data();
  }

  // Store transformed annotation.
  map<int, vector<AnnotationGroup> > all_anno; // batchsize中每一幅图像以及对应的标注
  int num_bboxes = 0;



 /*
  * ------------------------------------循环加载数据到batch------------------------------------------------------
  */
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
      // 对数据作distort
      distort_datum.CopyFrom(anno_datum);
      this->data_transformer_->DistortImage(anno_datum.datum(), distort_datum.mutable_datum());

      // distort的基础上做expand
      if (transform_param.has_expand_param()) {
        expand_datum = new AnnotatedDatum();
        this->data_transformer_->ExpandImage(distort_datum, expand_datum);
      } else {
        expand_datum = &distort_datum;
      }

    } else {
      // expand
      if (transform_param.has_expand_param()) {
        expand_datum = new AnnotatedDatum();
        this->data_transformer_->ExpandImage(anno_datum, expand_datum);
      } else {
        expand_datum = &anno_datum;
      }
    }

    //对数据做sample
    AnnotatedDatum* sampled_datum = NULL;
    bool has_sampled = false;

    if (batch_samplers_.size() > 0) {

      LOG(INFO) << "--------------------------------batch_samplers_:";

      // Generate sampled bboxes from expand_datum.
      vector<NormalizedBBox> sampled_bboxes;
      GenerateBatchSamples(*expand_datum, batch_samplers_, &sampled_bboxes);

      if (sampled_bboxes.size() > 0) {
        // Randomly pick a sampled bbox and crop the expand_datum.
        int rand_idx = caffe_rng_rand() % sampled_bboxes.size();
        sampled_datum = new AnnotatedDatum();
        this->data_transformer_->CropImage(*expand_datum, sampled_bboxes[rand_idx], sampled_datum);
        has_sampled = true;
      } else {
        sampled_datum = expand_datum;
      }

    } else {
      LOG(INFO) << "--------------------------------expand_datum:";
      sampled_datum = expand_datum;
    }
    CHECK(sampled_datum != NULL);

    timer.Start();
    vector<int> shape = this->data_transformer_->InferBlobShape(sampled_datum->datum());

    // todo  checkfiled  datum_channels > 0 (0 vs. 0)
//    LOG(INFO) << "datum_label" <<sampled_datum->datum_label();
    vector<int> shape_label = this->data_transformer_->InferBlobShape(sampled_datum->datum_label()); // added by pesong

    if (transform_param.has_resize_param()) {
          if (transform_param.resize_param().resize_mode() == ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {

            this->transformed_data_.Reshape(shape);
            batch->data_.Reshape(shape);

            this->transformed_label_img_.Reshape(shape); // added by pesong
            batch->label_img_.Reshape(shape); // added by pesong

            top_data = batch->data_.mutable_cpu_data();
            top_label_img = batch->label_img_.mutable_cpu_data();  // added by pesong
          } else {
            CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
                  shape.begin() + 1));
          }
    } else {
      CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
            shape.begin() + 1));
    }



//     Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);

    int offset_label_img = batch->label_img_.offset(item_id);   //added by pesong
    this->transformed_label_img_.set_cpu_data(top_label_img + offset_label_img);

    vector<AnnotationGroup> transformed_anno_vec;
    if (this->output_labels_) {

      if (has_anno_type_) {
        // Make sure all data have same annotation type.
        CHECK(sampled_datum->has_type()) << "Some datum misses AnnotationType.";
        if (anno_data_param.has_anno_type()) {
          sampled_datum->set_type(anno_type_);
        } else {
          CHECK_EQ(anno_type_, sampled_datum->type()) << "Different AnnotationType.";
        }

        // Transform datum and annotation_group at the same time
        transformed_anno_vec.clear();

         //!!!开始调用data_transformer.cpp line594
          //!!! Transform the cv::image into blob.
        this->data_transformer_->Transform(*sampled_datum, &(this->transformed_data_), &transformed_anno_vec);

          // todo   datum_label() Check failed: channels == datum_channels (3 vs. 0)
        this->data_transformer_->Transform(sampled_datum->datum_label(), &(this->transformed_label_img_));

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

INSTANTIATE_CLASS(AnnotatedDataWithSegLayer);
REGISTER_LAYER_CLASS(AnnotatedDataWithSeg);

}  // namespace caffe
