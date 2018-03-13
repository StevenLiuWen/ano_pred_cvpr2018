import tensorflow as tf
import os

from models import generator, discriminator, flownet, initialize_flownet
from loss_functions import intensity_loss, gradient_loss
from utils import DataLoader, load, save, psnr_error
from constant import const


os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = const.GPU

dataset_name = const.DATASET
train_folder = const.TRAIN_FOLDER
test_folder = const.TEST_FOLDER

batch_size = const.BATCH_SIZE
iterations = const.ITERATIONS
num_his = const.NUM_HIS
height, width = 256, 256
flow_height, flow_width = const.FLOW_HEIGHT, const.FLOW_WIDTH

l_num = const.L_NUM
alpha_num = const.ALPHA_NUM
lam_lp = const.LAM_LP
lam_gdl = const.LAM_GDL
lam_adv = const.LAM_ADV
lam_flow = const.LAM_FLOW
adversarial = (lam_adv != 0)

summary_dir = const.SUMMARY_DIR
snapshot_dir = const.SNAPSHOT_DIR


print(const)

# define dataset
with tf.name_scope('dataset'):
    train_loader = DataLoader(train_folder, resize_height=height, resize_width=width)
    train_dataset = train_loader(batch_size=batch_size, time_steps=num_his, num_pred=1)

    train_it = train_dataset.make_one_shot_iterator()
    train_videos_clips_tensor = train_it.get_next()
    train_videos_clips_tensor.set_shape([batch_size, height, width, 3*(num_his + 1)])

    train_inputs = train_videos_clips_tensor[..., 0:num_his*3]
    train_gt = train_videos_clips_tensor[..., -3:]

    print('train inputs = {}'.format(train_inputs))
    print('train prediction gt = {}'.format(train_gt))

    test_loader = DataLoader(test_folder, resize_height=height, resize_width=width)
    test_dataset = test_loader(batch_size=batch_size, time_steps=num_his, num_pred=1)
    test_it = test_dataset.make_one_shot_iterator()
    test_videos_clips_tensor = test_it.get_next()
    test_videos_clips_tensor.set_shape([batch_size, height, width, 3*(num_his + 1)])

    test_inputs = test_videos_clips_tensor[..., 0:num_his*3]
    test_gt = test_videos_clips_tensor[..., -3:]

    print('test inputs = {}'.format(test_inputs))
    print('test prediction gt = {}'.format(test_gt))

# define training generator function
with tf.variable_scope('generator', reuse=None):
    print('training = {}'.format(tf.get_variable_scope().name))
    train_outputs = generator(train_inputs, layers=4, output_channel=3)
    train_psnr_error = psnr_error(gen_frames=train_outputs, gt_frames=train_gt)

# define testing generator function
with tf.variable_scope('generator', reuse=True):
    print('testing = {}'.format(tf.get_variable_scope().name))
    test_outputs = generator(test_inputs, layers=4, output_channel=3)
    test_psnr_error = psnr_error(gen_frames=test_outputs, gt_frames=test_gt)


# define intensity loss
if lam_lp != 0:
    lp_loss = intensity_loss(gen_frames=train_outputs, gt_frames=train_gt, l_num=l_num)
else:
    lp_loss = tf.constant(0.0, dtype=tf.float32)


# define gdl loss
if lam_gdl != 0:
    gdl_loss = gradient_loss(gen_frames=train_outputs, gt_frames=train_gt, alpha=alpha_num)
else:
    gdl_loss = tf.constant(0.0, dtype=tf.float32)


# define flow loss
if lam_flow != 0:
    train_gt_flow = flownet(input_a=train_inputs[..., -3:], input_b=train_gt,
                            height=flow_height, width=flow_width, reuse=None)
    train_pred_flow = flownet(input_a=train_inputs[..., -3:], input_b=train_outputs,
                              height=flow_height, width=flow_width, reuse=True)
    flow_loss = tf.reduce_mean(tf.abs(train_gt_flow - train_pred_flow))
else:
    flow_loss = tf.constant(0.0, dtype=tf.float32)


# define adversarial loss
if adversarial:
    with tf.variable_scope('discriminator', reuse=None):
        real_logits, real_outputs = discriminator(inputs=train_gt)
    with tf.variable_scope('discriminator', reuse=True):
        fake_logits, fake_outputs = discriminator(inputs=train_outputs)

    print('real_outputs = {}'.format(real_outputs))
    print('fake_outputs = {}'.format(fake_outputs))

    adv_loss = tf.reduce_mean(tf.square(fake_outputs - 1) / 2)
    dis_loss = tf.reduce_mean(tf.square(real_outputs - 1) / 2) + tf.reduce_mean(tf.square(fake_outputs) / 2)
else:
    adv_loss = tf.constant(0.0, dtype=tf.float32)
    dis_loss = tf.constant(0.0, dtype=tf.float32)


with tf.name_scope('training'):
    g_loss = tf.add_n([lp_loss * lam_lp, gdl_loss * lam_gdl, adv_loss * lam_adv, flow_loss * lam_flow], name='g_loss')

    g_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='g_step')
    g_lrate = tf.train.piecewise_constant(g_step, boundaries=const.LRATE_G_BOUNDARIES, values=const.LRATE_G)
    g_optimizer = tf.train.AdamOptimizer(learning_rate=g_lrate, name='g_optimizer')
    g_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    g_train_op = g_optimizer.minimize(g_loss, global_step=g_step, var_list=g_vars, name='g_train_op')

    if adversarial:
        # training discriminator
        d_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='d_step')
        d_lrate = tf.train.piecewise_constant(d_step, boundaries=const.LRATE_D_BOUNDARIES, values=const.LRATE_D)
        d_optimizer = tf.train.AdamOptimizer(learning_rate=d_lrate, name='g_optimizer')
        d_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

        d_train_op = d_optimizer.minimize(dis_loss, global_step=d_step, var_list=d_vars, name='d_optimizer')
    else:
        d_step = None
        d_lrate = None
        d_train_op = None

# add all to summaries
tf.summary.scalar(tensor=train_psnr_error, name='train_psnr_error')
tf.summary.scalar(tensor=test_psnr_error, name='test_psnr_error')
tf.summary.scalar(tensor=g_loss, name='g_loss')
tf.summary.scalar(tensor=adv_loss, name='adv_loss')
tf.summary.scalar(tensor=dis_loss, name='dis_loss')
tf.summary.image(tensor=train_outputs, name='train_outputs')
tf.summary.image(tensor=train_gt, name='train_gt')
tf.summary.image(tensor=test_outputs, name='test_outputs')
tf.summary.image(tensor=test_gt, name='test_gt')
summary_op = tf.summary.merge_all()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # summaries
    summary_writer = tf.summary.FileWriter(summary_dir, graph=sess.graph)

    # initialize weights
    sess.run(tf.global_variables_initializer())
    print('Init successfully!')

    if lam_flow != 0:
        # initialize flownet
        initialize_flownet(sess, const.FLOWNET_CHECKPOINT)

    # tf saver
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)
    restore_var = [v for v in tf.global_variables()]
    loader = tf.train.Saver(var_list=restore_var)
    if os.path.isdir(snapshot_dir):
        ckpt = tf.train.get_checkpoint_state(snapshot_dir)
        if ckpt and ckpt.model_checkpoint_path:
            load(loader, sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found.')
    else:
        load(loader, sess, snapshot_dir)

    _step, _loss, _summaries = 0, None, None
    while _step < iterations:
        try:
            if adversarial:
                print('Training discriminator...')
                _, _d_lr, _d_step, _dis_loss = sess.run([d_train_op, d_lrate, d_step, dis_loss])
            else:
                _d_step = 0
                _d_lr = 0
                _dis_loss = 0

            print('Training generator...')
            _, _g_lr, _step, _lp_loss, _gdl_loss, _adv_loss, _flow_loss, _g_loss, _train_psnr, _summaries = sess.run(
                [g_train_op, g_lrate, g_step, lp_loss, gdl_loss, adv_loss, flow_loss, g_loss, train_psnr_error, summary_op])

            if _step % 10 == 0:
                print('DiscriminatorModel: Step {} | Global Loss: {:.6f}, lr = {:.6f}'.format(_d_step, _dis_loss, _d_lr))
                print('GeneratorModel : Step {}, lr = {:.6f}'.format(_step, _g_lr))
                print('                 Global      Loss : ', _g_loss)
                print('                 intensity   Loss : ({:.4f} * {:.4f} = {:.4f})'.format(_lp_loss, lam_lp, _lp_loss * lam_lp))
                print('                 gradient    Loss : ({:.4f} * {:.4f} = {:.4f})'.format( _gdl_loss, lam_gdl, _gdl_loss * lam_gdl))
                print('                 adversarial Loss : ({:.4f} * {:.4f} = {:.4f})'.format(_adv_loss, lam_adv, _adv_loss * lam_adv))
                print('                 flownet     Loss : ({:.4f} * {:.4f} = {:.4f})'.format(_flow_loss, lam_flow, _flow_loss * lam_flow))
                print('                 PSNR  Error      : ', _train_psnr)
            if _step % 100 == 0:
                summary_writer.add_summary(_summaries, global_step=_step)
                print('Save summaries...')

            if _step % 1000 == 0:
                save(saver, sess, snapshot_dir, _step)

        except tf.errors.OutOfRangeError:
            print('Finish successfully!')
            save(saver, sess, snapshot_dir, _step)
            break
