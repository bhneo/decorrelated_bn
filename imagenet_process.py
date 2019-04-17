import tensorflow as tf
import cv2
import numpy as np
import os
from multiprocessing import Process, Queue
import sys
import time
import random
import math

max_num = 1000  # max record number in one file
train_path = 'Imagenet/Imagenet/data/'  # the folder stroes the train images
valid_path = 'Imagenet/Imagenet/validation/'  # the folder stroes the validation images
cores = 4  # number of CPU cores to process

# Imagenet图片都保存在/data目录下，里面有1000个子目录，获取这些子目录的名字
classes = os.listdir(train_path)

# 构建一个字典，Key是目录名，value是类名0-999
labels_dict = {}
for i in range(len(classes)):
    labels_dict[classes[i]] = i

# 构建训练集文件列表，里面的每个元素是路径名+图片文件名+类名
images_labels_list = []
for i in range(len(classes)):
    path = train_path + classes[i] + '/'
    images_files = os.listdir(path)
    label = str(labels_dict[classes[i]])
    for image_file in images_files:
        images_labels_list.append(path + ',' + image_file + ',' + classes[i])
random.shuffle(images_labels_list)

# 读取验证集的图片对应的类名标签文件
valid_classes = []
with open('imagenet_2012_validation_synset_labels.txt', 'r') as f:
    valid_classes = [line.strip() for line in f.readlines()]
# 构建验证集文件列表，里面的每个元素是路径名+图片文件名+类名
valid_images_labels_list = []
valid_images_files = os.listdir(valid_path)
for file_item in valid_images_files:
    number = int(file_item[15:23]) - 1
    valid_images_labels_list.append(valid_path + ',' + file_item + ',' + valid_classes[number])


# 把图像数据和标签转换为TRRECORD的格式
def make_example(image, height, width, label, bbox, text):
    colorspace = b'RGB'
    channels = 3
    img_format = b'JPEG'
    return tf.train.Example(features=tf.train.Features(feature={
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'channels': tf.train.Feature(int64_list=tf.train.Int64List(value=[channels])),
        'colorspace': tf.train.Feature(bytes_list=tf.train.BytesList(value=[colorspace])),
        'img_format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_format])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'bbox_xmin': tf.train.Feature(float_list=tf.train.FloatList(value=bbox[0])),
        'bbox_xmax': tf.train.Feature(float_list=tf.train.FloatList(value=bbox[2])),
        'bbox_ymin': tf.train.Feature(float_list=tf.train.FloatList(value=bbox[1])),
        'bbox_ymax': tf.train.Feature(float_list=tf.train.FloatList(value=bbox[3])),
        'text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[text]))
    }))


# 读取bbox文件
bbox_list = {}
with open('bbox_train.csv', 'r') as bboxfile:
    records = bboxfile.readlines()
    for record in records:
        fields = record.strip().split(',')
        filename = fields[0][:-4]
        xmin = [float(x) for x in fields[1].split(';')]
        ymin = [float(x) for x in fields[2].split(';')]
        xmax = [float(x) for x in fields[3].split(';')]
        ymax = [float(x) for x in fields[4].split(';')]
        bbox_list[filename] = [xmin, ymin, xmax, ymax]

    # 读取Labels的描述
labels_text = {}
with open('imagenet_metadata.txt', 'r') as metafile:
    records = metafile.readlines()
    for record in records:
        fields = record.strip().split('\t')
        label = fields[0]
        text = fields[1]
        labels_text[label] = text


# 这个函数用来生成TFRECORD文件，第一个参数是列表，每个元素是图片文件名加类名，第二个参数是写入的目录名
# 第三个参数是文件名的起始序号，第四个参数是队列名称，用于和父进程发送消息
def gen_tfrecord(trainrecords, targetfolder, startnum, queue):
    tfrecords_file_num = startnum
    file_num = 0
    total_num = len(trainrecords)
    pid = os.getpid()
    queue.put((pid, file_num))
    writer = tf.python_io.TFRecordWriter(targetfolder + "train_" + str(tfrecords_file_num) + ".tfrecord")
    for record in trainrecords:
        file_num += 1
        fields = record.split(',')
        img = cv2.imread(fields[0] + fields[1])
        height, width, _ = img.shape
        img_jpg = cv2.imencode('.jpg', img)[1].tobytes()
        label = labels_dict[fields[2]]
        bbox = []
        try:
            bbox = bbox_list[fields[1][:-5]]
        except KeyError:
            bbox = [[], [], [], []]
        text = labels_text[fields[2]]
        ex = make_example(img_jpg, height, width, label, bbox, text.encode())
        writer.write(ex.SerializeToString())
        # 每写入100条记录，向父进程发送消息，报告进度
        if file_num % 100 == 0:
            queue.put((pid, file_num))
        if file_num % max_num == 0:
            writer.close()
            tfrecords_file_num += 1
            writer = tf.python_io.TFRecordWriter(targetfolder + "train_" + str(tfrecords_file_num) + ".tfrecord")
    writer.close()
    queue.put((pid, file_num))


# 这个函数用来多进程生成TFRECORD文件，第一个参数是要处理的图片的文件名列表，第二个参数是需要用的CPU核心数
# 第三个参数写入的文件目录名
def process_in_queues(fileslist, cores, targetfolder):
    total_files_num = len(fileslist)
    each_process_files_num = int(total_files_num / cores)
    files_for_process_list = []
    for i in range(cores - 1):
        files_for_process_list.append(fileslist[i * each_process_files_num:(i + 1) * each_process_files_num])
    files_for_process_list.append(fileslist[(cores - 1) * each_process_files_num:])
    files_number_list = [len(l) for l in files_for_process_list]

    each_process_tffiles_num = math.ceil(each_process_files_num / max_num)

    queues_list = []
    processes_list = []
    for i in range(cores):
        queues_list.append(Queue())
        # queue = Queue()
        processes_list.append(Process(target=gen_tfrecord,
                                      args=(files_for_process_list[i], targetfolder,
                                            each_process_tffiles_num * i + 1, queues_list[i],)))

    for p in processes_list:
        Process.start(p)

    # 父进程循环查询队列的消息，并且每0.5秒更新一次
    while (True):
        try:
            total = 0
            progress_str = ''
            for i in range(cores):
                msg = queues_list[i].get()
                total += msg[1]
                progress_str += 'PID' + str(msg[0]) + ':' + str(msg[1]) + '/' + str(files_number_list[i]) + '|'
            progress_str += '\r'
            print(progress_str, end='')
            if total == total_files_num:
                for p in processes_list:
                    p.terminate()
                    p.join()
                break
            time.sleep(0.5)
        except:
            break
    return total


if __name__ == '__main__':
    print('Start processing train data using %i CPU cores:' % cores)
    starttime = time.time()
    total_processed = process_in_queues(images_labels_list, cores, targetfolder='train_tf/')
    endtime = time.time()
    print('\nProcess finish, total process %i images in %i seconds' % (total_processed, int(endtime - starttime)))

    print('Start processing validation data using %i CPU cores:' % cores)
    starttime = time.time()
    total_processed = process_in_queues(valid_images_labels_list, cores, targetfolder='valid_tf/')
    endtime = time.time()
    print('\nProcess finish, total process %i images, using %i seconds' % (total_processed, int(endtime - starttime)))
