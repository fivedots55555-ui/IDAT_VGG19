"""Compare frozen VGG19 with a fine-tuned last block on one fixed split."""
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
from sklearn.model_selection import train_test_split

tf.keras.utils.set_random_seed(42)
OUT = ROOT / 'results' / 'independent_comparison'
OUT.mkdir(parents=True, exist_ok=True)
images = sorted(p for p in (ROOT / 'data/dataset').rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
classes = sorted({p.parent.name for p in images})
labels = np.array([classes.index(p.parent.name) for p in images])
data = np.array([cv2.resize(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), (224, 224)) for p in images], dtype=np.float32) / 255.0
train_val, test = train_test_split(np.arange(len(images)), test_size=0.20, random_state=42, stratify=labels)
train, val = train_test_split(train_val, test_size=0.25, random_state=42, stratify=labels[train_val])
split_names = {int(i): name for name, ids in [('train', train), ('validation', val), ('test', test)] for i in ids}
with (OUT / 'split.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.writer(stream)
    writer.writerow(['path', 'label', 'split', 'sha256'])
    for i, path in enumerate(images):
        writer.writerow([path.relative_to(ROOT).as_posix(), classes[labels[i]], split_names[i], hashlib.sha256(path.read_bytes()).hexdigest()])

one_hot = tf.keras.utils.to_categorical(labels, 2)
base = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base.trainable = False
head = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))(base.output)
head = tf.keras.layers.Flatten()(head)
head = tf.keras.layers.Dense(64, activation='relu')(head)
head = tf.keras.layers.Dropout(0.5)(head)
head = tf.keras.layers.Dense(2, activation='softmax')(head)
model = tf.keras.Model(base.input, head)
augmentation = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15, fill_mode='nearest')
summary = {}


def fit_and_evaluate(name, epochs, learning_rate):
    folder = OUT / name
    folder.mkdir(exist_ok=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(
        augmentation.flow(data[train], one_hot[train], batch_size=5, seed=42),
        validation_data=(data[val], one_hot[val]), epochs=epochs, verbose=2,
        callbacks=[tf.keras.callbacks.CSVLogger(str(folder / 'history.csv')),
                   tf.keras.callbacks.ModelCheckpoint(str(folder / 'best.keras'), monitor='val_loss', save_best_only=True)],
    )
    best = tf.keras.models.load_model(folder / 'best.keras')
    probabilities = best.predict(data[test], batch_size=5, verbose=0)
    predicted = probabilities.argmax(axis=1)
    report = classification_report(labels[test], predicted, labels=[0, 1], target_names=classes, output_dict=True, zero_division=0)
    (folder / 'metrics_test.json').write_text(json.dumps(report, indent=2))
    with (folder / 'predictions_test.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['path', 'true', 'predicted', *['prob_' + c for c in classes]])
        for i, pred, probs in zip(test, predicted, probabilities):
            writer.writerow([images[i].relative_to(ROOT).as_posix(), classes[labels[i]], classes[pred], *probs.tolist()])
    for metric in ['accuracy', 'loss']:
        fig, ax = plt.subplots()
        ax.plot(history.history[metric], label='train')
        ax.plot(history.history['val_' + metric], label='validation')
        ax.set(xlabel='Epoch', ylabel=metric)
        ax.legend()
        fig.savefig(folder / f'{metric}.png', dpi=180, bbox_inches='tight')
        plt.close(fig)
    ConfusionMatrixDisplay(confusion_matrix(labels[test], predicted, labels=[0, 1]), display_labels=classes).plot()
    plt.savefig(folder / 'confusion_matrix_test.png', dpi=180, bbox_inches='tight')
    plt.close()
    summary[name] = {'accuracy': report['accuracy'], 'defect_recall': report['badcylinder']['recall'],
                     'defect_f1': report['badcylinder']['f1-score'], 'best_validation_epoch': int(np.argmin(history.history['val_loss'])) + 1}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({name: summary[name]}), flush=True)


fit_and_evaluate('frozen_baseline', 50, 0.001)
base.trainable = True
for layer in base.layers:
    layer.trainable = layer.name.startswith('block5_')
fit_and_evaluate('block5_finetuned', 20, 0.00001)
(OUT / 'config.json').write_text(json.dumps({
    'split': 'stratified 60% train, 20% validation, 20% test; random_state 42',
    'baseline': 'frozen ImageNet VGG19, same head and preprocessing as public notebook, Adam 0.001, 50 epochs',
    'improvement': 'start from final frozen baseline; unfreeze VGG19 block 5; Adam 0.00001, 20 epochs',
    'selection': 'lowest validation loss checkpoint; evaluate test once per model',
    'limitation': 'Images were split individually; images from one print or capture series may span splits.'
}, indent=2))
