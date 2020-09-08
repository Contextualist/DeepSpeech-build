#===config.py===
from deepspeech_training.util.config import *
from deepspeech_training.util.config import initialize_globals as __initialize_globals

def initialize_globals():
    __initialize_globals()
    ConfigSingleton._config.n_input = 512 # XXX: for MFCC it's 26, but for wav2vec it's 512

from deepspeech_training.util import config
config.initialize_globals = initialize_globals


#===checkpints.py===
from deepspeech_training.util.checkpoints import *

def _load_checkpoint(session, checkpoint_path, allow_drop_layers):
    # Load the checkpoint and put all variables into loading list
    # we will exclude variables we do not wish to load and then
    # we will initialize them instead
    ckpt = tfv1.train.load_checkpoint(checkpoint_path)
    vars_in_ckpt = frozenset(ckpt.get_variable_to_shape_map().keys())
    load_vars = set(tfv1.global_variables())
    init_vars = set()

    # We explicitly allow the learning rate variable to be missing for backwards
    # compatibility with older checkpoints.
    lr_var = set(v for v in load_vars if v.op.name == 'learning_rate')
    if lr_var and ('learning_rate' not in vars_in_ckpt or FLAGS.force_initialize_learning_rate):
        assert len(lr_var) <= 1
        load_vars -= lr_var
        init_vars |= lr_var

    if FLAGS.load_cudnn:
        # Initialize training from a CuDNN RNN checkpoint
        # Identify the variables which we cannot load, and set them
        # for initialization
        missing_vars = set()
        for v in load_vars:
            if v.op.name not in vars_in_ckpt:
                log_warn('CUDNN variable not found: %s' % (v.op.name))
                missing_vars.add(v)
                init_vars.add(v)

        load_vars -= init_vars

        # Check that the only missing variables (i.e. those to be initialised)
        # are the Adam moment tensors, if they aren't then we have an issue
        missing_var_names = [v.op.name for v in missing_vars]
        if any('Adam' not in v for v in missing_var_names):
            log_error('Tried to load a CuDNN RNN checkpoint but there were '
                      'more missing variables than just the Adam moment '
                      'tensors. Missing variables: {}'.format(missing_var_names))
            sys.exit(1)

    if allow_drop_layers and FLAGS.drop_source_layers > 0:
        # This transfer learning approach requires supplying
        # the layers which we exclude from the source model.
        # Say we want to exclude all layers except for the first one,
        # then we are dropping five layers total, so: drop_source_layers=5
        # If we want to use all layers from the source model except
        # the last one, we use this: drop_source_layers=1
        if FLAGS.drop_source_layers >= 6:
            log_warn('The checkpoint only has 6 layers, but you are trying to drop '
                     'all of them or more than all of them. Continuing and '
                     'dropping only 5 layers.')
            FLAGS.drop_source_layers = 5

        dropped_layers = ['2', '3', 'lstm', '5', '6'][-1 * int(FLAGS.drop_source_layers):]
        dropped_layers = ['1', *dropped_layers] # XXX: drop first layer
        # Initialize all variables needed for DS, but not loaded from ckpt
        for v in load_vars:
            if any(layer in v.op.name for layer in dropped_layers):
                init_vars.add(v)
        load_vars -= init_vars

    for v in sorted(load_vars, key=lambda v: v.op.name):
        log_info('Loading variable from checkpoint: %s' % (v.op.name))
        v.load(ckpt.get_tensor(v.op.name), session=session)

    for v in sorted(init_vars, key=lambda v: v.op.name):
        log_info('Initializing variable: %s' % (v.op.name))
        session.run(v.initializer)

from deepspeech_training.util import checkpoints
checkpoints._load_checkpoint = _load_checkpoint


#===audio.py===
from deepspeech_training.util.audio import *

AUDIO_TYPE_HDF5 = "application/x-hdf5"

def Sample__init(self, audio_type, raw_data, audio_format=None, sample_id=None):
    self.audio_type = audio_type
    self.audio_format = audio_format
    self.sample_id = sample_id
    if audio_type in SERIALIZABLE_AUDIO_TYPES:
        self.audio = raw_data if isinstance(raw_data, io.BytesIO) else io.BytesIO(raw_data)
        self.duration = read_duration(audio_type, self.audio)
    else:
        self.audio = raw_data
        if self.audio_format is None:
            raise ValueError('For audio type "{}" parameter "audio_format" is mandatory'.format(self.audio_type))
        if audio_type == AUDIO_TYPE_PCM:
            self.duration = get_pcm_duration(len(self.audio), self.audio_format)
        elif audio_type == AUDIO_TYPE_NP:
            self.duration = get_np_duration(len(self.audio), self.audio_format)
        elif audio_type == AUDIO_TYPE_HDF5:
            self.duration = 0 # XXX: can we do that?
        else:
            raise ValueError('Unsupported audio type: {}'.format(self.audio_type))

from deepspeech_training.util import audio
audio.LOADABLE_AUDIO_EXTENSIONS['.h5context'] = AUDIO_TYPE_HDF5 # Sample.audio_type
audio.Sample.__init__ = Sample__init


#===sample_collections.py===
from deepspeech_training.util.sample_collections import *

def load_sample(filename, label=None):
    ext = os.path.splitext(filename)[1].lower()
    audio_type = get_audio_type_from_extension(ext)
    if audio_type is None:
        raise ValueError('Unknown audio type extension "{}"'.format(ext))
    if audio_type == AUDIO_TYPE_HDF5:
        import h5py
        with h5py.File(filename, 'r') as hd5f_file:
            audio_data = hd5f_file['features'][()] # XXX: wav2vec feature field
    else:
        with open(filename, 'rb') as audio_file:
            audio_data = audio_file.read()
    if label is None:
        return Sample(audio_type, audio_data, sample_id=filename)
    return LabeledSample(audio_type, audio_data, label, sample_id=filename)

from deepspeech_training.util import sample_collections
sample_collections.load_sample = load_sample


#===augmentations.py===
def apply_sample_augmentations(samples, *args, **kwargs): # XXX: forcefully disable agmentation, as it tries to convert the audio to PCM type
    return samples

from deepspeech_training.util import augmentations
augmentations.apply_sample_augmentations = apply_sample_augmentations


#===feeding.py===
from deepspeech_training.util.feeding import *
from deepspeech_training.util.feeding import audio_to_features as __audio_to_features
import sys

def audio_to_features(samples, sample_rate, transcript=None, clock=0.0, train_phase=False, augmentations=None, sample_id=None):
    def _hdf5():
        #tf.print('WARNING: Disable MFCC featuration on HDF5 file', sample_id, output_stream=sys.stdout)
        feat = tf.reshape(samples, [-1, Config.n_input])
        return feat, tf.shape(input=feat)[0]
    def _non_hdf5():
        #tf.print('WARNING: Perform MFCC featuration on file', sample_id, output_stream=sys.stdout)
        __audio_to_features(samples, sample_rate, transcript, clock, train_phase, augmentations, sample_id)
    #r = tf.cond( # XXX: I give up, this does not run
    #    tf.strings.regex_full_match([sample_id], r'.*\.h5context$'),
    #    lambda: _hdf5(),
    #    lambda: _non_hdf5(),
    #)
    #return r
    return _hdf5()

def audiofile_to_features(wav_filename, clock=0.0, train_phase=False, augmentations=None):
    import tensorflow as tf
    @tf.function # encapsulate non-tensor-ops into a tensor-op
    def load_hdf5():
        import h5py
        with h5py.File(wav_filename, 'r') as hd5f_file:
            return hd5f_file['features'][()] # XXX: wav2vec feature field
    samples = load_hdf5()
    return audio_to_features(samples,
                             -1,
                             clock=clock,
                             train_phase=train_phase,
                             augmentations=augmentations,
                             sample_id=wav_filename)

from deepspeech_training.util import feeding
feeding.audio_to_features = audio_to_features
feeding.audiofile_to_features = audiofile_to_features

