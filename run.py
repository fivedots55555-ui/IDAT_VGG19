"""Reproduce the public AM3d notebook using its VGG19 switch."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('KERAS_HOME', str(ROOT / '.keras'))


def download_data():
    tree = json.loads((ROOT / 'author/dataset_tree.json').read_text(encoding='utf-8-sig'))
    entries = [x for x in tree['tree'] if x['type'] == 'blob' and x['path'].startswith('dataset/')]

    def fetch(entry):
        relative = entry['path']
        target = ROOT / 'data' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            content = target.read_bytes()
            digest = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
            if digest == entry['sha']:
                return
        url = 'https://raw.githubusercontent.com/mahsan2/AMdataset/' + tree['sha'] + '/' + relative
        for attempt in range(3):
            try:
                content = urllib.request.urlopen(url, timeout=90).read()
                digest = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
                if digest != entry['sha']:
                    raise ValueError('Download checksum mismatch: ' + relative)
                target.write_bytes(content)
                return
            except Exception:
                if attempt == 2:
                    raise

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(fetch, entries))
    print(f'Downloaded and verified {len(entries)} files', flush=True)


def train(epochs):
    import cv2
    import numpy as np
    import tensorflow as tf
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

    tf.keras.utils.set_random_seed(42)
    output = ROOT / 'results' / f'baseline_{epochs}epochs'
    output.mkdir(parents=True, exist_ok=True)
    config = dict(model='VGG19', weights='imagenet', image_size=224, normalization='RGB / 255',
                  frozen_backbone=True, pooling='AveragePooling2D(4,4)', dense=64, dropout=0.5,
                  output='2-way softmax', loss='binary_crossentropy', optimizer='Adam',
                  learning_rate=0.001, batch_size=5, epochs=epochs, seed=42,
                  split='80/20 random_state=42, without stratification', rotation_range=15,
                  evaluation='20% holdout reused for validation, following author notebook',
                  differences=['Sorted file paths; original notebook file order unavailable',
                               'Use all batches per epoch to avoid Keras 3 iterator exhaustion with floor steps',
                               'Explicit global seed 42; original notebook does not fix all seeds',
                               'Public dataset subset; exact paper dataset correspondence unverified'])
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    images = sorted(p for p in (ROOT / 'data/dataset').rglob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png'])
    classes = sorted({p.parent.name for p in images})
    if len(classes) != 2:
        raise ValueError(f'Expected two classes, found {classes}')
    labels = np.array([classes.index(p.parent.name) for p in images])
    data = np.array([cv2.resize(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), (224, 224)) for p in images], dtype=np.float32) / 255.0
    train_ids, test_ids = train_test_split(np.arange(len(images)), test_size=0.20, random_state=42)
    train_set = set(train_ids.tolist())
    with (output / 'split.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['path', 'label', 'split', 'sha256'])
        for index, path in enumerate(images):
            writer.writerow([path.relative_to(ROOT).as_posix(), classes[labels[index]], 'train' if index in train_set else 'holdout', hashlib.sha256(path.read_bytes()).hexdigest()])
    (output / 'environment.json').write_text(json.dumps(dict(tensorflow=tf.__version__, numpy=np.__version__, classes=classes, devices=[str(d) for d in tf.config.list_physical_devices()]), indent=2))
    backbone = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    backbone.trainable = False
    head = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))(backbone.output)
    head = tf.keras.layers.Flatten()(head)
    head = tf.keras.layers.Dense(64, activation='relu')(head)
    head = tf.keras.layers.Dropout(0.5)(head)
    predictions = tf.keras.layers.Dense(2, activation='softmax')(head)
    model = tf.keras.Model(backbone.input, predictions)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    with (output / 'architecture.txt').open('w') as stream:
        model.summary(print_fn=lambda line: stream.write(line + '\n'))
    augmentation = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15, fill_mode='nearest')
    one_hot = tf.keras.utils.to_categorical(labels, num_classes=2)
    callbacks = [tf.keras.callbacks.CSVLogger(str(output / 'history.csv')),
                 tf.keras.callbacks.ModelCheckpoint(str(output / 'best_validation.keras'), monitor='val_loss', save_best_only=True)]
    history = model.fit(augmentation.flow(data[train_ids], one_hot[train_ids], batch_size=5),
                        validation_data=(data[test_ids], one_hot[test_ids]),
                        epochs=epochs, callbacks=callbacks, verbose=2)
    model.save(output / 'final.keras')
    probabilities = model.predict(data[test_ids], batch_size=5)
    predicted = probabilities.argmax(axis=1)
    report = classification_report(labels[test_ids], predicted, labels=[0, 1], target_names=classes, output_dict=True, zero_division=0)
    (output / 'metrics_final.json').write_text(json.dumps(report, indent=2))
    with (output / 'predictions_final.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['path', 'true', 'predicted', *['prob_' + c for c in classes]])
        for index, prediction, probability in zip(test_ids, predicted, probabilities):
            writer.writerow([images[index].relative_to(ROOT), classes[labels[index]], classes[prediction], *probability.tolist()])
    for metric in ['accuracy', 'loss']:
        fig, ax = plt.subplots()
        ax.plot(history.history[metric], label='train')
        ax.plot(history.history['val_' + metric], label='holdout')
        ax.set(xlabel='Epoch (zero-based)', ylabel=metric)
        ax.legend()
        fig.savefig(output / (metric + '.png'), dpi=180, bbox_inches='tight')
        plt.close(fig)
    ConfusionMatrixDisplay(confusion_matrix(labels[test_ids], predicted, labels=[0, 1]), display_labels=classes).plot()
    plt.savefig(output / 'confusion_matrix_final.png', dpi=180, bbox_inches='tight')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    if args.download:
        download_data()
    else:
        train(args.epochs)
