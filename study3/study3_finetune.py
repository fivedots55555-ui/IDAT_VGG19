"""Fine-tune VGG19 block 5 after the official-preprocessing Study Three run."""
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
study = ROOT / 'results/study3'
out = study / 'block5_finetuned'
out.mkdir(exist_ok=True)
rows = list(csv.DictReader((study / 'split.csv').open(encoding='utf-8')))
classes = ['defected', 'no_defected']
labels = np.array([classes.index(row['label']) for row in rows])
ids = {name: np.array([i for i, row in enumerate(rows) if row['split'] == name]) for name in ['train', 'validation', 'test']}
paths = [ROOT / row['path'] for row in rows]
raw = np.array([cv2.resize(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB), (224, 224)) for path in paths], dtype=np.float32)
processed = tf.keras.applications.vgg19.preprocess_input(raw.copy())
one_hot = tf.keras.utils.to_categorical(labels, 2)
model = tf.keras.models.load_model(study / 'vgg_preprocess/best.keras')
for layer in model.layers:
    if layer.name.startswith('block'):
        layer.trainable = layer.name.startswith('block5_')
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])
augmentation = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15, fill_mode='nearest',
    preprocessing_function=tf.keras.applications.vgg19.preprocess_input,
)
history = model.fit(
    augmentation.flow(raw[ids['train']], one_hot[ids['train']], batch_size=30, seed=42),
    validation_data=(processed[ids['validation']], one_hot[ids['validation']]),
    epochs=10, verbose=2,
    callbacks=[tf.keras.callbacks.CSVLogger(str(out / 'history.csv')),
               tf.keras.callbacks.ModelCheckpoint(str(out / 'best.keras'), monitor='val_loss', save_best_only=True)],
)
best_epoch = int(np.argmin(history.history['val_loss'])) + 1
best = tf.keras.models.load_model(out / 'best.keras')
probabilities = best.predict(processed[ids['test']], batch_size=30, verbose=0)
predicted = probabilities.argmax(axis=1)
report = classification_report(labels[ids['test']], predicted, labels=[0, 1], target_names=classes, output_dict=True, zero_division=0)
(out / 'metrics_test.json').write_text(json.dumps(report, indent=2))
with (out / 'predictions_test.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.writer(stream)
    writer.writerow(['path', 'true', 'predicted', 'prob_defected', 'prob_no_defected'])
    for i, prediction, probability in zip(ids['test'], predicted, probabilities):
        writer.writerow([rows[i]['path'], classes[labels[i]], classes[prediction], *probability.tolist()])
for metric in ['accuracy', 'loss']:
    fig, ax = plt.subplots()
    ax.plot(history.history[metric], label='train')
    ax.plot(history.history['val_' + metric], label='validation')
    ax.set(xlabel='Fine-tuning epoch', ylabel=metric)
    ax.legend()
    fig.savefig(out / f'{metric}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
ConfusionMatrixDisplay(confusion_matrix(labels[ids['test']], predicted, labels=[0, 1]), display_labels=classes).plot()
plt.savefig(out / 'confusion_matrix_test.png', dpi=180, bbox_inches='tight')
plt.close()
(out / 'config.json').write_text(json.dumps({
    'initial_weights': 'best validation checkpoint from study3/vgg_preprocess',
    'trainable_vgg_layers': 'block5 only', 'learning_rate': 0.00001,
    'batch_size': 30, 'epochs': 10, 'selection': 'lowest validation loss checkpoint',
    'best_validation_epoch': best_epoch,
    'limitation': 'Exploratory follow-up after the test set was inspected in prior experiments.'
}, indent=2))
print(json.dumps({'accuracy': report['accuracy'], 'defect_recall': report['defected']['recall'],
                  'defect_f1': report['defected']['f1-score'], 'best_validation_epoch': best_epoch}), flush=True)
