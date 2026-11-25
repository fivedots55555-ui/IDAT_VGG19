"""VGG19 baseline and one-input-preprocessing comparison for paper Study Three."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('KERAS_HOME', str(ROOT / '.keras'))

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['paper', 'vgg_preprocess'], required=True)
args = parser.parse_args()
tf.keras.utils.set_random_seed(42)
root_out = ROOT / 'results/study3'
out = root_out / args.mode
out.mkdir(parents=True, exist_ok=True)

paths = sorted(p for p in (ROOT / 'data/study3').rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
classes = ['defected', 'no_defected']
labels = np.array([classes.index(p.parent.name) for p in paths])
expected = {'defected': (528, 78, 153), 'no_defected': (561, 81, 156)}
assert {name: int(sum(labels == i)) for i, name in enumerate(classes)} == {
    name: sum(counts) for name, counts in expected.items()
}

split_file = root_out / 'split.csv'
if split_file.exists():
    saved = list(csv.DictReader(split_file.open(encoding='utf-8')))
    assert [row['path'] for row in saved] == [p.relative_to(ROOT).as_posix() for p in paths]
    splits = {name: np.array([i for i, row in enumerate(saved) if row['split'] == name]) for name in ['train', 'validation', 'test']}
else:
    rng = np.random.RandomState(42)
    memberships = {}
    for class_index, class_name in enumerate(classes):
        indices = np.flatnonzero(labels == class_index)
        rng.shuffle(indices)
        train_count, validation_count, _ = expected[class_name]
        for name, group in [('train', indices[:train_count]),
                            ('validation', indices[train_count:train_count + validation_count]),
                            ('test', indices[train_count + validation_count:])]:
            memberships.update({int(index): name for index in group})
    with split_file.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['path', 'label', 'split', 'sha256'])
        for index, path in enumerate(paths):
            writer.writerow([path.relative_to(ROOT).as_posix(), classes[labels[index]], memberships[index], hashlib.sha256(path.read_bytes()).hexdigest()])
    splits = {name: np.array([i for i in range(len(paths)) if memberships[i] == name]) for name in ['train', 'validation', 'test']}

raw = np.array([cv2.resize(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB), (224, 224)) for path in paths], dtype=np.float32)
if args.mode == 'paper':
    processed = raw / 255.0
    augmentation = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15, fill_mode='nearest')
    train_data = processed[splits['train']]
else:
    processed = tf.keras.applications.vgg19.preprocess_input(raw.copy())
    augmentation = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15, fill_mode='nearest',
        preprocessing_function=tf.keras.applications.vgg19.preprocess_input,
    )
    train_data = raw[splits['train']]
one_hot = tf.keras.utils.to_categorical(labels, 2)

base = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base.trainable = False
head = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))(base.output)
head = tf.keras.layers.Flatten()(head)
head = tf.keras.layers.Dense(64, activation='relu')(head)
head = tf.keras.layers.Dropout(0.5)(head)
head = tf.keras.layers.Dense(2, activation='softmax')(head)
model = tf.keras.Model(base.input, head)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(
    augmentation.flow(train_data, one_hot[splits['train']], batch_size=30, seed=42),
    validation_data=(processed[splits['validation']], one_hot[splits['validation']]),
    epochs=30, verbose=2,
    callbacks=[tf.keras.callbacks.CSVLogger(str(out / 'history.csv')),
               tf.keras.callbacks.ModelCheckpoint(str(out / 'best.keras'), monitor='val_loss', save_best_only=True)],
)
best_epoch = int(np.argmin(history.history['val_loss'])) + 1
best = tf.keras.models.load_model(out / 'best.keras')
probabilities = best.predict(processed[splits['test']], batch_size=30, verbose=0)
predicted = probabilities.argmax(axis=1)
report = classification_report(labels[splits['test']], predicted, labels=[0, 1], target_names=classes, output_dict=True, zero_division=0)
(out / 'metrics_test.json').write_text(json.dumps(report, indent=2))
with (out / 'predictions_test.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.writer(stream)
    writer.writerow(['path', 'true', 'predicted', 'prob_defected', 'prob_no_defected'])
    for i, prediction, probability in zip(splits['test'], predicted, probabilities):
        writer.writerow([paths[i].relative_to(ROOT).as_posix(), classes[labels[i]], classes[prediction], *probability.tolist()])
for metric in ['accuracy', 'loss']:
    fig, ax = plt.subplots()
    ax.plot(history.history[metric], label='train')
    ax.plot(history.history['val_' + metric], label='validation')
    ax.set(xlabel='Epoch', ylabel=metric)
    ax.legend()
    fig.savefig(out / f'{metric}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
ConfusionMatrixDisplay(confusion_matrix(labels[splits['test']], predicted, labels=[0, 1]), display_labels=classes).plot()
plt.savefig(out / 'confusion_matrix_test.png', dpi=180, bbox_inches='tight')
plt.close()
(out / 'config.json').write_text(json.dumps({
    'source': 'Ahsan et al., Expert Systems with Applications 253 (2024) 124293, Table 1 and Section 4',
    'data': 'Kaggle 3D-Printer Defected Dataset, public GitHub mirror pingkung/3D-Printer-Defected-Dataset',
    'class_counts': expected,
    'split': 'class-wise shuffled with seed 42 to match paper counts; exact author image membership unavailable',
    'model': 'ImageNet VGG19 frozen convolutional base, average pooling 4x4, flatten, dense 64 ReLU, dropout 0.5, softmax 2',
    'optimizer': 'Adam, learning rate 0.001', 'batch_size': 30, 'epochs': 30,
    'preprocessing': 'RGB / 255' if args.mode == 'paper' else 'tf.keras.applications.vgg19.preprocess_input',
    'augmentation': 'rotation_range=15, fill_mode=nearest on training images',
    'selection': 'lowest validation loss checkpoint', 'best_validation_epoch': best_epoch,
    'limitation': 'Paper reports mean and standard deviation; this is one deterministic split/run.'
}, indent=2))
print(json.dumps({'mode': args.mode, 'accuracy': report['accuracy'],
                  'defect_recall': report['defected']['recall'],
                  'defect_f1': report['defected']['f1-score'],
                  'best_validation_epoch': best_epoch}), flush=True)
