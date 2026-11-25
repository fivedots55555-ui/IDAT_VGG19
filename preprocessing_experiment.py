"""Test VGG19's documented ImageNet preprocessing on the existing split."""
import csv
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

tf.keras.utils.set_random_seed(42)
out = ROOT / 'results' / 'official_preprocessing'
out.mkdir(parents=True, exist_ok=True)
split_file = ROOT / 'results' / 'independent_comparison' / 'split.csv'
rows = list(csv.DictReader(split_file.open(encoding='utf-8')))
classes = sorted({row['label'] for row in rows})
images = [ROOT / row['path'] for row in rows]
labels = np.array([classes.index(row['label']) for row in rows])
raw = np.array([cv2.resize(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB), (224, 224)) for path in images], dtype=np.float32)
one_hot = tf.keras.utils.to_categorical(labels, num_classes=2)
ids = {name: np.array([i for i, row in enumerate(rows) if row['split'] == name]) for name in ['train', 'validation', 'test']}
preprocess = tf.keras.applications.vgg19.preprocess_input
validation = preprocess(raw[ids['validation']].copy())
test = preprocess(raw[ids['test']].copy())

backbone = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
backbone.trainable = False
head = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))(backbone.output)
head = tf.keras.layers.Flatten()(head)
head = tf.keras.layers.Dense(64, activation='relu')(head)
head = tf.keras.layers.Dropout(0.5)(head)
head = tf.keras.layers.Dense(2, activation='softmax')(head)
model = tf.keras.Model(backbone.input, head)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
augmentation = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15, fill_mode='nearest', preprocessing_function=preprocess
)
history = model.fit(
    augmentation.flow(raw[ids['train']], one_hot[ids['train']], batch_size=5, seed=42),
    validation_data=(validation, one_hot[ids['validation']]),
    epochs=50, verbose=2,
    callbacks=[tf.keras.callbacks.CSVLogger(str(out / 'history.csv')),
               tf.keras.callbacks.ModelCheckpoint(str(out / 'best.keras'), monitor='val_loss', save_best_only=True)],
)
best_epoch = int(np.argmin(history.history['val_loss'])) + 1
best = tf.keras.models.load_model(out / 'best.keras')
probabilities = best.predict(test, batch_size=5, verbose=0)
predicted = probabilities.argmax(axis=1)
report = classification_report(labels[ids['test']], predicted, labels=[0, 1], target_names=classes, output_dict=True, zero_division=0)
(out / 'metrics_test.json').write_text(json.dumps(report, indent=2))
with (out / 'predictions_test.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.writer(stream)
    writer.writerow(['path', 'true', 'predicted', *['prob_' + c for c in classes]])
    for i, prediction, probability in zip(ids['test'], predicted, probabilities):
        writer.writerow([rows[i]['path'], classes[labels[i]], classes[prediction], *probability.tolist()])
for metric in ['accuracy', 'loss']:
    fig, ax = plt.subplots()
    ax.plot(history.history[metric], label='train')
    ax.plot(history.history['val_' + metric], label='validation')
    ax.set(xlabel='Epoch', ylabel=metric)
    ax.legend()
    fig.savefig(out / f'{metric}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
ConfusionMatrixDisplay(confusion_matrix(labels[ids['test']], predicted, labels=[0, 1]), display_labels=classes).plot()
plt.savefig(out / 'confusion_matrix_test.png', dpi=180, bbox_inches='tight')
plt.close()
(out / 'config.json').write_text(json.dumps({
    'change': 'VGG19 official preprocess_input applied after training rotation and to validation/test',
    'unchanged': 'same split, frozen backbone, head, Adam 0.001, batch size 5, 50 epochs',
    'best_validation_epoch': best_epoch,
    'source': 'https://keras.io/api/applications/vgg/vgg_preprocessing/',
    'limitation': 'This test set was already inspected for the earlier baseline and fine-tuning comparison.'
}, indent=2))
print(json.dumps({'accuracy': report['accuracy'], 'defect_recall': report['badcylinder']['recall'],
                  'defect_f1': report['badcylinder']['f1-score'], 'best_validation_epoch': best_epoch}), flush=True)
