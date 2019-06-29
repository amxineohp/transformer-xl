import logging, sys, os, inspect, json, numpy as np, gc
from copy import deepcopy
import tensorflow as tf
import mautil as mu
import math
from mautil.basic_model import InputFeature
from mautil import data_reader
from mautil.tf_models import TF
from tfxl.model import transformer
from tfxl import tpu_estimator
import util


SEED = 9527
gl = globals()


class TFXL(TF):
    cfg = deepcopy(TF.cfg)
    cfg.batch_reader = 'BatchedSeqBatchReader'
    cfg.batch_reader_cfg = {'process_mode': 'T', 'max_seq_len': 100}
    # gpu
    cfg.num_hosts = 1
    cfg.num_core_per_host = 8

    # Experiment (data/checkpoint/directory) config
    cfg.data_dir = "data" 
    cfg.record_info_dir = ""  
    cfg.corpus_info_path = ""  
    cfg.model_dir = None  # Estimator model_dir.
    cfg.do_train = True  # Whether to run training.
    cfg.do_eval = False  # Whether to run eval on the dev set.

    # Optimization config
    cfg.lr = 2.5e-4   # "Maximum learning rate."
    cfg.gradient_clip = 0.25   # "Gradient clipping value."
        # for cosine decay
    cfg.min_lr_ratio = 0.004   # "Minimum ratio learning rate."
    cfg.warmup_steps = 0   # "Number of steps for linear lr warmup."


        # Evaluation config
    cfg.do_test = False   # "Run on the test set."
    cfg.max_eval_batch = -1   # "Set -1 to turn off. Only used in test mode."
    cfg.do_eval_only = False   # "Run evaluation only."
    cfg.start_eval_steps = 10000   # "Which checkpoint to start with in `do_eval_only` mode."
    cfg.eval_split = "valid"  # Which data split to evaluate.")

        # Model config
    cfg.tgt_len = 70   # "Number of steps to predict"
    cfg.mem_len = 70   # "Number of steps to cache"
    cfg.same_length = False   # "Same length attention"
    cfg.clamp_len = -1   # "Clamp length"

    cfg.n_layer = 6   # "Number of layers."
    cfg.d_model = 500   # "Dimension of the model."
    cfg.d_embed = 500   # "Dimension of the embeddings."
    cfg.n_head = 10   # "Number of attention heads."
    cfg.d_head = 50   # "Dimension of each attention head."
    cfg.d_inner = 1000   # "Dimension of inner hidden size in positionwise feed-forward."
    cfg.dropout = 0.1   # "Dropout rate."
    cfg.dropatt = 0.1   # "Attention dropout rate."
    cfg.untie_r = False   # "untie r_w_bias and r_r_bias"

        # Adaptive Softmax / Embedding
    cfg.tie_weight = True   # "Tie embedding and softmax weight."
    cfg.div_val = 1   # "Divide the embedding size by this val for each bin"
    cfg.proj_share_all_but_first = False   # "True to share all but first projs, False not to share."
    cfg.proj_same_dim = True   # "Project the bin with the same dimension."

        # Parameter initialization
    cfg.init = "normal"  # "Initialization method."
    cfg.init_std = 0.02   # "Initialization std when init is normal."
    cfg.proj_init_std = 0.01   # "Initialization std for embedding projection."
    cfg.init_range = 0.1   # "Initialization std when init is uniform."


    cfg.input_perms = None
    cfg.target_perms = None
    cfg.head_target = None

    def __init__(self, name, cfg={}, batch_reader=None):
        self.cfg = deepcopy(self.cfg)
        if cfg['dataset'] == 'ptb':
            self.cfg.div_val = 1
            self.cfg.n_layer = 12
            self.cfg.d_model = 400
            self.cfg.d_embed = 400
            self.cfg.n_head = 8
            self.cfg.d_head = 50
            self.cfg.d_inner = 1000
            # Training
            self.cfg.dropout = 0.1
            self.cfg.dropatt = 0.0
            self.cfg.tgt_len = 50
            self.cfg.mem_len = 50
            self.cfg.batch_size = 16
            self.cfg.val_batch_size = 16
            self.cfg.proj_share_all_but_first = True
            self.cfg.tpu_loop_iterations = 50

        elif cfg['dataset'] == 'wt103':
            self.cfg.div_val = 1
            self.cfg.n_layer = 16
            self.cfg.d_model = 410
            self.cfg.d_embed = 410
            self.cfg.n_head = 10
            self.cfg.d_head = 41
            self.cfg.d_inner = 2100
            self.cfg.untie_r = True

            # Training
            self.cfg.lr = 0.00025
            self.cfg.dropout = 0.1
            self.cfg.dropatt = 0.0
            self.cfg.tgt_len = 150
            self.cfg.mem_len = 150
            self.cfg.batch_size = 16
            self.cfg.val_batch_size = 16
            self.cfg.lr_decay_step = 400000
            self.cfg.save_step = 4000
            self.cfg.proj_share_all_but_first = True
            self.cfg.tpu_loop_iterations = 1000

            self.cfg.test_tgt_len = 64
            self.cfg.test_mem_len = 640
            self.cfg.test_clamp_len = 400


        self.cfg.update(cfg)

        if self.cfg.debug:
            self.cfg.n_layer = 2
            self.cfg.d_model = 4
            self.cfg.d_embed = 4
            self.cfg.n_head = 2
            self.cfg.d_head = 2
            self.cfg.d_inner = 4
            self.cfg.batch_reader_cfg['max_seq_len'] = 10
            self.cfg.tpu_loop_iterations = 2

        self.cfg.batch_reader_cfg['max_seq_len'] = self.cfg.tgt_len + 1
        name = name + '_' + self.cfg.dataset

        super(TFXL, self).__init__(name, self.cfg.__dict__, batch_reader)

        tie_projs = [False for _ in range(len(self.cfg.cutoffs) + 1)]
        if self.cfg.proj_share_all_but_first:
            for i in range(1, len(tie_projs)):
                tie_projs[i] = True
        self.cfg.tie_projs = tie_projs
        if self.cfg.no_cutoffs:
            self.cfg.cutoffs = []

        self.tower_mems_np = [np.zeros([self.cfg.batch_size, self.cfg.mem_len, self.cfg.d_model], dtype=np.float32) for l in range(self.cfg.n_layer)]

    def pre_run_batch(self, batch, epoch=0, itr=0, global_step=0, is_training=True):
        batch = super(TFXL, self).pre_run_batch(batch, epoch=epoch, itr=itr, global_step=global_step, is_training=is_training)
        if self.cfg.mem_len >0:
            for i in range(self.cfg.n_layer):
                batch['mems_'+str(i)] = self.tower_mems_np[i]
        #batch['seqs'] = np.transpose(batch['seqs'], [1,0])
        return batch

    def _add_train_nodes(self):
        super(TFXL, self)._add_train_nodes()
        self.train_nodes['mems'] = self._new_mems
        self.validate_nodes['mems'] = self._new_mems

    def before_run(self, run_context):
        run_args = super(TFXL, self).before_run(run_context)
        feed_dict = run_args.feed_dict
        fetches = run_args.fetches

        if self.cfg.mem_len >0:
            for i in range(self.cfg.n_layer):
                feed_dict[getattr(self, 'mems_' + str(i))] = self.tower_mems_np[i]
        run_args = tf.train.SessionRunArgs(feed_dict=feed_dict, fetches=fetches)
        return run_args

    def after_run(self, run_context, run_values):
        super(TFXL, self).after_run(run_context, run_values)
        mems = run_values.results['mems']
        self.tower_mems_np = [np.transpose(m, [1, 0, 2]) for m in mems]

    def run(self, sess, batch, nodes):
        outputs = super(TFXL, self).run(sess, batch, nodes)
        self.tower_mems_np = [np.transpose(m, [1, 0, 2]) for m in outputs['mems']]
        return outputs

    def _set_model_fn_inputs(self, features, params):
        super(TFXL, self)._set_model_fn_inputs(features, params)
        if self.cfg.use_tpu:
            setattr(self, 'mems', params['cache'])

    def _get_cache_fn(self, mem_len):

        def cache_fn(batch_size):
            logging.info('cache batch_size is %s', batch_size)
            mems = []
            for l in range(self.cfg.n_layer):
                if mem_len > 0:
                    mems.append(
                        tf.zeros([mem_len, batch_size, self.cfg.d_model], dtype=tf.float32))
                else:
                    mems.append(tf.zeros([mem_len], dtype=tf.float32))

            return mems

        return cache_fn

    def _metric_fn(self, loss):
        perplexity = tf.exp(tf.reduce_mean(loss))
        bpc = tf.reduce_mean(loss) / tf.constant(math.log(2))
        return {
            "perplexity": tf.metrics.mean(perplexity),
            "bpc": tf.metrics.mean(bpc),
        }

    def _get_estimator_spec(self, mode, batch_size, hooks=None):
        if self.cfg.use_tpu:
            hooks = None
        output_spec = super(TFXL, self)._get_estimator_spec(mode, batch_size, hooks=hooks)
        if self.cfg.use_tpu:
            output_spec.cache = self._new_mems
        return output_spec

    def _get_estimator(self, model_fn, run_config, warm_start_from):
        if not self.cfg.use_tpu:
            return super(TFXL, self)._get_estimator(model_fn, run_config, warm_start_from)
        train_cache_fn = None
        if not self.cfg.only_validate:
            train_cache_fn = self._get_cache_fn(self.cfg.mem_len)
        eval_cache_fn = self._get_cache_fn(self.cfg.mem_len)
        estimator = tpu_estimator.TPUEstimator(
            use_tpu=self.cfg.use_tpu,
            model_fn=model_fn,
            params={'track_mean': True},
            train_cache_fn=train_cache_fn,
            eval_cache_fn=eval_cache_fn,
            config=run_config,
            train_batch_size=self.cfg.batch_size,
            eval_batch_size=self.cfg.val_batch_size,
            warm_start_from=warm_start_from)
        logging.info('customized tpu_estimator used ')
        return estimator

    def _get_tfrecord_spec(self, input_features):
        new_input_features = []
        for fea in input_features:
            if not fea.name.startswith('mems'):
                new_input_features.append(fea)
        return super(TFXL, self)._get_tfrecord_spec(new_input_features)

    def _add_main_graph(self):
        if self.cfg.mem_len >0:
            if not self.cfg.use_tpu:
                self.mems = []
                for i in range(self.cfg.n_layer):
                    self.mems.append(tf.transpose(getattr(self, 'mems_'+ str(i)), [1, 0, 2]))
        if self.cfg.init == "uniform":
            initializer = tf.initializers.random_uniform(
                minval=-self.cfg.init_range,
                maxval=self.cfg.init_range,
                seed=self.cfg.seed)
        elif self.cfg.init == "normal":
            initializer = tf.initializers.random_normal(
                stddev=self.cfg.init_std,
                seed=self.cfg.seed)
            proj_initializer = tf.initializers.random_normal(
                stddev=self.cfg.proj_init_std,
                seed=self.cfg.seed)
        self._loss, self._new_mems = transformer(
                dec_inp=tf.transpose(self.seqs[:, 0:-1], [1,0]),
                target=tf.transpose(self.seqs[:, 1:], [1,0]),
                mems=self.mems,
                n_token=self.cfg.n_token,
                n_layer=self.cfg.n_layer,
                d_model=self.cfg.d_model,
                d_embed=self.cfg.d_embed,
                n_head=self.cfg.n_head,
                d_head=self.cfg.d_head,
                d_inner=self.cfg.d_inner,
                dropout=self.cfg.dropout,
                dropatt=self.cfg.dropatt,
                initializer=initializer,
                proj_initializer=proj_initializer,
                is_training=self._training_plh,
                mem_len=self.cfg.mem_len,
                cutoffs=self.cfg.cutoffs,
                div_val=self.cfg.div_val,
                tie_projs=self.cfg.tie_projs,
                input_perms=self.cfg.input_perms,
                target_perms=self.cfg.target_perms,
                head_target=self.cfg.head_target,
                same_length=self.cfg.same_length,
                clamp_len=self.cfg.clamp_len,
                use_tpu=self.cfg.use_tpu,
                untie_r=self.cfg.untie_r,
                proj_same_dim=self.cfg.proj_same_dim)

    def _init_input_features(self):
        features = []
        features.append(InputFeature('seqs', [self.cfg.batch_size, self.cfg.tgt_len+1], tf.int64))
        if self.cfg.mem_len >0:
            for i in range(self.cfg.n_layer):
                features.append(InputFeature('mems_'+ str(i), [self.cfg.batch_size, self.cfg.mem_len, self.cfg.d_model], tf.float32))
        return features


class ArgParser(mu.TrainArgParser):
    @staticmethod
    def add_args(parser):
        super(ArgParser, ArgParser).add_args(parser)
        parser.add_argument("-tgt_len", "--tgt_len", type=int, help="tgt seq len")
        parser.add_argument("-mem_len", "--mem_len", type=int, help="mem len")
        parser.add_argument("-no_cutoffs", "--no_cutoffs", action="store_true", help="no cuttoffs")


def train(args):
    trainer = mu.training.Trainer('Trainer', SEED)
    data_dict, corpus_info = util.load_data('data/datasets', args.dataset, debug=args.debug)
    args.n_token = corpus_info['vocab_size']
    args.cutoffs = corpus_info['cutoffs']

    trainer.train_model(data_dict, args, gl, process_data=False)


if __name__ == '__main__':
    args = ArgParser.load_args()
    from imp import reload
    reload(logging)

    if args.debug:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(threadName)s %(message)s')
    else:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s:%(threadName)s %(message)s')

    if args.method_or_model in gl:
        if inspect.isfunction(gl[args.method_or_model]):
            gl[args.method_or_model](args)
        else:
            args.model_names = args.method_or_model
            train(args)
    else:
        logging.error('unknown method or model: %s', args.method_or_model)

