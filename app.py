import sys
import os
import json
import time
import math
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from PySide6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, 
                               QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                               QPlainTextEdit, QLineEdit, QProgressBar, QMessageBox,
                               QSpinBox, QDoubleSpinBox, QFormLayout, QComboBox,
                               QTableWidget, QTableWidgetItem, QTextBrowser,
                               QHeaderView, QSplitter, QFileDialog, QGroupBox,
                               QRadioButton, QFrame, QListWidget, QCheckBox,
                               QTextEdit)
from PySide6.QtCore import QThread, Signal, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.style as mpl_style
import io
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
import networkx as nx

mpl_style.use('dark_background')

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_steps):
        super().__init__()
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = self.base_lr * step / max(1.0, self.warmup_steps)
        decay_steps = max(1.0, self.total_steps - self.warmup_steps)
        decay = 0.5 * self.base_lr * (1 + tf.cos(np.pi * (step - self.warmup_steps) / decay_steps))
        return tf.where(step < self.warmup_steps, warmup, decay)

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, pos_type='learned'):
        super().__init__()
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_type = pos_type
        if pos_type == 'learned':
            self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        elif pos_type == 'sinusoidal':
            pe = np.zeros((maxlen, embed_dim))
            position = np.arange(0, maxlen, dtype=np.float32).reshape(-1, 1)
            div_term = np.exp(np.arange(0, embed_dim, 2).astype(np.float32) * (-np.log(10000.0) / embed_dim))
            pe[:, 0::2] = np.sin(position * div_term)
            pe[:, 1::2] = np.cos(position * div_term)
            self.pos_emb = tf.constant(pe[None, :, :], dtype=tf.float32)

    def call(self, x):
        x = self.token_emb(x)
        if self.pos_type == 'none':
            return x
        elif self.pos_type == 'learned':
            maxlen = tf.shape(x)[1]
            positions = tf.range(start=0, limit=maxlen, delta=1)
            return x + self.pos_emb(positions)
        else: 
            return x + self.pos_emb[:, :tf.shape(x)[1], :]

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, activation='relu', rate=0.1, use_pre_ln=False):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation=activation),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.last_attn_scores = None
        self.use_pre_ln = use_pre_ln

    def call(self, inputs, training=False, mask=None):
        if self.use_pre_ln:
            normed_x = self.layernorm1(inputs)
            attn_output, attn_scores = self.att(normed_x, normed_x, attention_mask=mask, return_attention_scores=True)
            self.last_attn_scores = attn_scores.numpy() if not training else None
            attn_output = self.dropout1(attn_output, training=training)
            out1 = inputs + attn_output
            
            normed_out1 = self.layernorm2(out1)
            ffn_output = self.ffn(normed_out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return out1 + ffn_output
        else:
            attn_output, attn_scores = self.att(inputs, inputs, attention_mask=mask, return_attention_scores=True)
            self.last_attn_scores = attn_scores.numpy() if not training else None
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)

class LLMManager:
    def __init__(self):
        self.raw_text = ""
        self.tokenizer = None
        self.vocab_size = 0
        self.train_dataset = None
        self.val_dataset = None
        self.model = None
        self.context_extractors = [] 
        self.transformer_blocks = []
        self.embedding_layer = None
        self.seq_length = 24
        self.config = {}
        self.token_freqs = []
        self.id_to_freq = {}
        self.doc_lengths = []
        self.bigram_probs = defaultdict(list)

    def load_data(self, dataset_name, custom_path=None):
        if custom_path:
            with open(custom_path, 'r', encoding='utf-8') as f:
                text = f.read()
            texts = [t for t in text.split('\n') if len(t) > 50][:2000]
            self.raw_text = "\n".join(texts)
            return len(texts), "Custom Text"
        elif dataset_name == "wikitext":
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            texts = [t for t in ds['text'] if len(t) > 50][:2000]
            self.raw_text = "\n".join(texts)
            return len(texts), "WikiText"
        elif dataset_name == "shakespeare":
            ds = load_dataset("tiny_shakespeare", split="train")
            texts = ds['text'][0].split('\n')
            texts = [t for t in texts if len(t) > 50][:2000]
            self.raw_text = "\n".join(texts)
            return len(texts), "Shakespeare"

    def build_tokenizer(self):
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizers = Whitespace()
        trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "<bos>", "<eos>"], vocab_size=2000)
        chunks = [self.raw_text[i:i+5000] for i in range(0, len(self.raw_text), 5000)]
        self.tokenizer.train_from_iterator(chunks, trainer)
        self.vocab_size = self.tokenizer.get_vocab_size()
        
        bos_id = self.tokenizer.token_to_id("<bos>")
        eos_id = self.tokenizer.token_to_id("<eos>")
        
        tokens = []
        texts = self.raw_text.split('\n')
        self.doc_lengths = []
        for t in texts:
            if len(t.strip()) == 0: continue
            enc = self.tokenizer.encode(t)
            self.doc_lengths.append(len(enc.ids))
            tokens.extend([bos_id] + enc.ids + [eos_id])
        
        token_counts = Counter(tokens)
        self.token_freqs = sorted(token_counts.values(), reverse=True)
        self.id_to_freq = np.zeros(self.vocab_size)
        for tid, count in token_counts.items():
            self.id_to_freq[tid] = count
            
        self.bigram_probs = defaultdict(list)
        for i in range(len(tokens) - 1):
            self.bigram_probs[tokens[i]].append(tokens[i+1])
        
        def split_input_target(chunk):
            input_text = chunk[:-1]
            target_text = chunk[1:]
            return input_text, target_text

        sequences = tf.data.Dataset.from_tensor_slices(tokens)
        sequences = sequences.batch(self.seq_length + 1, drop_remainder=True)
        dataset = sequences.map(split_input_target)
        
        dataset_size = len(dataset)
        val_size = int(0.1 * dataset_size)
        self.val_dataset = dataset.take(val_size).batch(32).prefetch(tf.data.AUTOTUNE)
        self.train_dataset = dataset.skip(val_size).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
        
        num_chars = len(self.raw_text)
        num_tokens = len(tokens)
        ratio = num_chars / num_tokens if num_tokens > 0 else 0
        
        tokenizer_json = json.loads(self.tokenizer.to_str())
        merges = tokenizer_json["model"]["merges"]
        
        return self.vocab_size, num_chars, num_tokens, ratio, merges

    def generate_bigram_text(self, seed_text, length=20):
        enc = self.tokenizer.encode(seed_text)
        if not enc.ids: return "Invalid seed."
        curr_id = enc.ids[-1]
        text = seed_text
        
        for _ in range(length):
            if curr_id not in self.bigram_probs or not self.bigram_probs[curr_id]:
                break
            next_id = np.random.choice(self.bigram_probs[curr_id])
            text += " " + self.tokenizer.decode([next_id])
            if next_id == self.tokenizer.token_to_id("<eos>"):
                break
            curr_id = next_id
        return text

    def build_model(self, embed_dim, num_heads, ff_dim, num_layers, activation, optimizer_name, clip_norm, use_pre_ln, pos_type, dropout_rate, weight_decay, epochs, warmup_steps, batch_size, freeze_emb):
        self.config = {
            'embed_dim': embed_dim, 'num_heads': num_heads, 
            'ff_dim': ff_dim, 'num_layers': num_layers, 'activation': activation,
            'optimizer': optimizer_name, 'clip_norm': clip_norm, 'use_pre_ln': use_pre_ln,
            'pos_type': pos_type, 'dropout_rate': dropout_rate, 'weight_decay': weight_decay,
            'epochs': epochs, 'warmup_steps': warmup_steps, 'batch_size': batch_size, 'freeze_emb': freeze_emb
        }
        
        # Re-batch datasets with new batch size
        def split_input_target(chunk):
            input_text = chunk[:-1]
            target_text = chunk[1:]
            return input_text, target_text

        tokens = []
        texts = self.raw_text.split('\n')
        for t in texts:
            if len(t.strip()) == 0: continue
            enc = self.tokenizer.encode(t)
            tokens.extend([self.tokenizer.token_to_id("<bos>")] + enc.ids + [self.tokenizer.token_to_id("<eos>")])
        
        sequences = tf.data.Dataset.from_tensor_slices(tokens)
        sequences = sequences.batch(self.seq_length + 1, drop_remainder=True)
        dataset = sequences.map(split_input_target)
        
        dataset_size = len(dataset)
        val_size = int(0.1 * dataset_size)
        self.val_dataset = dataset.take(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        self.train_dataset = dataset.skip(val_size).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        seq_len = self.seq_length
        inputs = tf.keras.layers.Input(shape=(seq_len,))
        
        self.embedding_layer = TokenAndPositionEmbedding(seq_len, self.vocab_size, embed_dim, pos_type)
        
        if freeze_emb:
            self.embedding_layer.token_emb.trainable = False
            
        x = self.embedding_layer(inputs)
        
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        causal_mask = tf.expand_dims(causal_mask, axis=0)
        
        self.transformer_blocks = []
        self.context_extractors = [] 
        
        self.context_extractors.append(tf.keras.Model(inputs=inputs, outputs=x))
        
        for _ in range(num_layers):
            block = TransformerBlock(embed_dim, num_heads, ff_dim, activation, rate=dropout_rate, use_pre_ln=use_pre_ln)
            self.transformer_blocks.append(block)
            x = block(x, mask=causal_mask)
            self.context_extractors.append(tf.keras.Model(inputs=inputs, outputs=x))
            
        outputs = tf.keras.layers.Dense(self.vocab_size)(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        steps_per_epoch = len(self.train_dataset)
        total_steps = steps_per_epoch * epochs
        lr_schedule = WarmupCosineDecay(base_lr=1e-3, total_steps=total_steps, warmup_steps=warmup_steps)
        
        if optimizer_name == "AdamW":
            opt = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=weight_decay, clipnorm=clip_norm)
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=clip_norm)
            
        self.model.compile(optimizer=opt, 
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        
        string_io = io.StringIO()
        self.model.summary(print_fn=lambda x: string_io.write(x + "\n"))
        
        params = {}
        params['Token Emb'] = self.vocab_size * embed_dim
        if pos_type == 'learned': params['Pos Emb'] = seq_len * embed_dim
        else: params['Pos Emb'] = 0
        
        att_params = 4 * (embed_dim * embed_dim) + 4 * embed_dim
        ffn_params = (embed_dim * ff_dim) + ff_dim + (ff_dim * embed_dim) + embed_dim
        ln_params = 4 * embed_dim
        
        params['Attention (All)'] = att_params * num_layers
        params['FFN (All)'] = ffn_params * num_layers
        params['LayerNorm (All)'] = ln_params * num_layers
        params['Output Layer'] = embed_dim * self.vocab_size + self.vocab_size
        
        total_params = sum(params.values())
        trainable_params = np.sum([np.prod(v.shape) for v in self.model.trainable_weights])
        
        return self.model.count_params(), string_io.getvalue(), params, total_params, int(trainable_params)

class TrainingThread(QThread):
    update_log = Signal(str)
    update_progress = Signal(int, dict)
    training_done = Signal()

    def __init__(self, llm_manager, epochs):
        super().__init__()
        self.llm = llm_manager
        self.epochs = epochs

    def run(self):
        class LogCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self_inner, epoch, logs=None):
                metrics = {}
                metrics['loss'] = logs.get('loss', 0)
                metrics['val_loss'] = logs.get('val_loss', 0)
                metrics['acc'] = logs.get('accuracy', 0)
                metrics['lr'] = logs.get('lr', 0)
                metrics['ppl'] = float(np.exp(min(metrics['loss'], 20)))
                
                top1, top3, top5, total = 0, 0, 0, 0
                try:
                    for x_val, y_val in self.llm.val_dataset.take(1):
                        preds = self.model(x_val, training=False)
                        top_k_preds = tf.math.top_k(preds, k=5).indices.numpy()
                        y_val_np = y_val.numpy()
                        for i in range(y_val_np.shape[0]):
                            for j in range(y_val_np.shape[1]):
                                true_id = y_val_np[i, j]
                                if true_id == 0: continue 
                                top_k = top_k_preds[i, j]
                                if true_id in top_k[:1]: top1 += 1
                                if true_id in top_k[:3]: top3 += 1
                                if true_id in top_k[:5]: top5 += 1
                                total += 1
                        break
                except:
                    pass
                
                metrics['top1'] = top1 / total if total > 0 else 0
                metrics['top3'] = top3 / total if total > 0 else 0
                metrics['top5'] = top5 / total if total > 0 else 0
                
                metrics['grad_norm'] = 0.0
                metrics['emb_grads'] = []
                metrics['layer_norms'] = []
                try:
                    weights = self.model.trainable_weights
                    for x_batch, y_batch in self.llm.train_dataset.take(1):
                        with tf.GradientTape() as tape:
                            preds = self.model(x_batch, training=True)
                            loss_val = self.model.compiled_loss(y_batch, preds)
                        grads = tape.gradient(loss_val, weights)
                        metrics['grad_norm'] = float(tf.linalg.global_norm(grads).numpy())
                        
                        block_names = set()
                        for w in weights:
                            if 'transformer_block' in w.name:
                                block_names.add(w.name.split('/')[0])
                        
                        layer_norms = [('Embed', 0.0)]
                        emb_g = [g for g, w in zip(grads, weights) if 'token_emb' in w.name and g is not None]
                        if emb_g: layer_norms[-1] = ('Embed', float(tf.linalg.global_norm(emb_g).numpy()))
                        
                        for bn in sorted(block_names):
                            bg = [g for g, w in zip(grads, weights) if bn in w.name and g is not None]
                            if bg: layer_norms.append((bn, float(tf.linalg.global_norm(bg).numpy())))
                            
                        out_g = [g for g, w in zip(grads, weights) if 'dense' in w.name and 'transformer' not in w.name and g is not None]
                        if out_g: layer_norms.append(('Output', float(tf.linalg.global_norm(out_g).numpy())))
                        metrics['layer_norms'] = layer_norms
                        
                        for g, w in zip(grads, weights):
                            if 'token_emb' in w.name and g is not None:
                                metrics['emb_grads'] = g.numpy().flatten().tolist()
                                break
                        break
                except:
                    pass
                
                self.update_log.emit(f"Epoch {epoch+1}/{self.epochs} - Loss: {metrics['loss']:.4f} - Val Loss: {metrics['val_loss']:.4f} - Top-1: {metrics['top1']:.2%} - Top-5: {metrics['top5']:.2%} - PPL: {metrics['ppl']:.2f}")
                self.update_progress.emit(int(((epoch+1)/self.epochs)*100), metrics)

        self.update_log.emit("Starting training...")
        self.llm.model.fit(self.llm.train_dataset, validation_data=self.llm.val_dataset, 
                           epochs=self.epochs, callbacks=[LogCallback()], verbose=0)
        self.update_log.emit("Training Complete!")
        self.training_done.emit()

class GenerationThread(QThread):
    update_text = Signal(str, bool) # text, is_final
    update_probs = Signal(list, list, list) # tokens, top_10_probs, top_50_probs
    update_metrics = Signal(float, float)
    update_traj = Signal(int, float)
    update_margin = Signal(int, float)
    update_beam_tree = Signal(list) 
    generation_done = Signal(list, object)

    def __init__(self, llm_manager, seed_text, next_words, temp, top_k, top_p, strategy, rep_penalty, beam_width=3):
        super().__init__()
        self.llm = llm_manager
        self.seed_text = seed_text
        self.next_words = next_words
        self.temp = temp
        self.top_k = top_k
        self.top_p = top_p
        self.strategy = strategy
        self.rep_penalty = rep_penalty
        self.beam_width = beam_width
        self.is_running = True
        self.eos_id = self.llm.tokenizer.token_to_id("<eos>")

    def stop(self):
        self.is_running = False

    def get_token_ids(self, text):
        enc = self.llm.tokenizer.encode(text)
        token_ids = enc.ids
        if len(token_ids) > self.llm.seq_length:
            token_ids = token_ids[-self.llm.seq_length:]
        if len(token_ids) < self.llm.seq_length:
            padding = [0] * (self.llm.seq_length - len(token_ids))
            token_ids = padding + token_ids
        return token_ids

    def apply_repetition_penalty(self, logits, generated_ids):
        for id in set(generated_ids):
            if logits[id] < 0:
                logits[id] *= self.rep_penalty
            else:
                logits[id] /= self.rep_penalty
        return logits

    def run(self):
        start_time = time.time()
        generated_tokens = 0
        
        if self.strategy == "Sampling":
            self.run_sampling(start_time)
        elif self.strategy == "Greedy":
            self.run_greedy(start_time)
        elif self.strategy == "Beam Search":
            self.run_beam_search(start_time)
        
        final_ids = self.get_token_ids(self.seed_text)
        final_logits = self.llm.model.predict(np.array([final_ids]), verbose=0)[0]
        self.generation_done.emit(final_ids, final_logits)

    def run_sampling(self, start_time):
        generated_text = self.seed_text
        history_ids = self.llm.tokenizer.encode(generated_text).ids
        
        for step in range(self.next_words):
            if not self.is_running: break
            token_ids = self.get_token_ids(generated_text)
            predictions = self.llm.model.predict(np.array([token_ids]), verbose=0)
            logits = predictions[0, -1, :] / self.temp
            
            logits = self.apply_repetition_penalty(logits, history_ids)
            
            if self.top_k > 0:
                top_k_vals = tf.math.top_k(logits, k=self.top_k)[0].numpy()[-1]
                logits[logits < top_k_vals] = -float('Inf')
            if self.top_p < 1.0:
                sorted_indices = np.argsort(logits)[::-1]
                cumulative_probs = np.cumsum(tf.nn.softmax(logits).numpy()[sorted_indices])
                sorted_indices_to_remove = cumulative_probs > self.top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1]
                sorted_indices_to_remove[0] = False
                logits[sorted_indices[sorted_indices_to_remove]] = -float('Inf')
                
            probs = tf.nn.softmax(logits).numpy()
            probs[0] = 0; probs[1] = 0
            probs = probs / np.sum(probs)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            
            top_10_idx = np.argsort(probs)[-10:][::-1]
            top_50_idx = np.argsort(probs)[-50:][::-1]
            self.update_probs.emit([self.llm.tokenizer.decode([int(i)]) for i in top_10_idx], probs[top_10_idx].tolist(), probs[top_50_idx].tolist())
            
            sorted_logits = np.sort(predictions[0, -1, :])[::-1]
            margin = float(sorted_logits[0] - sorted_logits[1])
            
            pred_id = np.random.choice(len(probs), p=probs)
            if pred_id == self.eos_id:
                generated_text += "<eos>"
                self.update_text.emit(generated_text, True)
                break
                
            chosen_prob = probs[pred_id]
            history_ids.append(pred_id)
            generated_text += self.llm.tokenizer.decode([pred_id]) + " "
            self.seed_text = generated_text
            self.update_text.emit(generated_text, False)
            self.update_traj.emit(step+1, chosen_prob)
            self.update_margin.emit(step+1, margin)
            
            generated_tokens += 1
            tps = generated_tokens / (time.time() - start_time)
            self.update_metrics.emit(tps, entropy)
            self.msleep(100)

    def run_greedy(self, start_time):
        generated_text = self.seed_text
        history_ids = self.llm.tokenizer.encode(generated_text).ids
        
        for step in range(self.next_words):
            if not self.is_running: break
            token_ids = self.get_token_ids(generated_text)
            predictions = self.llm.model.predict(np.array([token_ids]), verbose=0)
            logits = predictions[0, -1, :]
            
            logits = self.apply_repetition_penalty(logits, history_ids)
            probs = tf.nn.softmax(logits).numpy()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            
            top_10_idx = np.argsort(probs)[-10:][::-1]
            top_50_idx = np.argsort(probs)[-50:][::-1]
            self.update_probs.emit([self.llm.tokenizer.decode([int(i)]) for i in top_10_idx], probs[top_10_idx].tolist(), probs[top_50_idx].tolist())
            
            sorted_logits = np.sort(logits)[::-1]
            margin = float(sorted_logits[0] - sorted_logits[1])
            
            pred_id = np.argmax(logits)
            if pred_id == self.eos_id:
                generated_text += "<eos>"
                self.update_text.emit(generated_text, True)
                break
                
            chosen_prob = probs[pred_id]
            history_ids.append(pred_id)
            generated_text += self.llm.tokenizer.decode([pred_id]) + " "
            self.seed_text = generated_text
            self.update_text.emit(generated_text, False)
            self.update_traj.emit(step+1, chosen_prob)
            self.update_margin.emit(step+1, margin)
            
            generated_tokens += 1
            tps = generated_tokens / (time.time() - start_time)
            self.update_metrics.emit(tps, entropy)
            self.msleep(100)

    def run_beam_search(self, start_time):
        beams = [(self.seed_text, 0.0, self.llm.tokenizer.encode(self.seed_text).ids)]
        edges = []
        
        for step in range(self.next_words):
            if not self.is_running: break
            candidates = []
            for text, score, hist_ids in beams:
                token_ids = self.get_token_ids(text)
                predictions = self.llm.model.predict(np.array([token_ids]), verbose=0)
                logits = predictions[0, -1, :]
                logits = self.apply_repetition_penalty(logits, hist_ids)
                log_probs = tf.nn.log_softmax(logits).numpy()
                
                top_k_idx = np.argsort(log_probs)[-self.beam_width:][::-1]
                for idx in top_k_idx:
                    new_text = text + " " + self.llm.tokenizer.decode([int(idx)])
                    new_score = score + log_probs[idx]
                    new_hist = hist_ids + [int(idx)]
                    candidates.append((new_text, new_score, new_hist))
                    edges.append((text, new_text, float(log_probs[idx])))
                    
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:self.beam_width]
            self.update_text.emit(beams[0][0], False)
            self.seed_text = beams[0][0]
            self.update_beam_tree.emit(edges)
            
            best_text = beams[0][0]
            token_ids = self.get_token_ids(best_text)
            predictions = self.llm.model.predict(np.array([token_ids]), verbose=0)
            probs = tf.nn.softmax(predictions[0, -1, :]).numpy()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            
            top_10_idx = np.argsort(probs)[-10:][::-1]
            top_50_idx = np.argsort(probs)[-50:][::-1]
            self.update_probs.emit([self.llm.tokenizer.decode([int(i)]) for i in top_10_idx], probs[top_10_idx].tolist(), probs[top_50_idx].tolist())
            
            sorted_logits = np.sort(predictions[0, -1, :])[::-1]
            margin = float(sorted_logits[0] - sorted_logits[1])
            chosen_prob = probs[np.argmax(predictions[0, -1, :])]
            self.update_traj.emit(step+1, chosen_prob)
            self.update_margin.emit(step+1, margin)
            
            generated_tokens += 1
            tps = generated_tokens / (time.time() - start_time)
            self.update_metrics.emit(tps, entropy)
            self.msleep(150)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLM Research Lab - Ultimate NLP Academy")
        self.resize(1600, 1100)
        self.setStyleSheet("""
            QWidget { background-color: #1e1e1e; color: #d4d4d4; font-size: 14px; }
            QPushButton { background-color: #0288d1; border: none; padding: 8px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #03a9f4; }
            QPushButton:disabled { background-color: #555; color: #888; }
            QPushButton#StopBtn { background-color: #e91e63; }
            QPushButton#StopBtn:hover { background-color: #ff4081; }
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab { background: #2d2d2d; padding: 8px 20px; border: 1px solid #444; border-bottom: none; }
            QTabBar::tab:selected { background: #1e1e1e; border-bottom: 2px solid #0288d1; }
            QPlainTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background-color: #252526; border: 1px solid #444; padding: 4px; border-radius: 2px; }
            QTextEdit { background-color: #252526; border: 1px solid #444; padding: 4px; border-radius: 2px; }
            QTableWidget { background-color: #252526; gridline-color: #444; }
            QHeaderView::section { background-color: #333; border: 1px solid #444; }
            QTextBrowser { background-color: #252526; border: 1px solid #444; }
            QGroupBox { border: 1px solid #444; border-radius: 6px; margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QFrame#MetricFrame { background-color: #252526; border: 1px solid #444; border-radius: 4px; }
            QListWidget { background-color: #252526; border: 1px solid #444; }
            QListWidget::item:selected { background-color: #0288d1; color: white; }
            QCheckBox { spacing: 8px; }
        """)
        
        self.llm = LLMManager()
        self.training_thread = None
        self.generation_thread = None
        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(100)
        main_layout.addWidget(self.console)

        self.log("Application started. Begin with the Pipeline Guide tab.")

        self.setup_tab_guide()
        self.setup_tab_concepts()
        self.setup_tab_data()
        self.setup_tab_tokenizer()
        self.setup_tab_model()
        self.setup_tab_train()
        self.setup_tab_generate()
        self.setup_tab_embeddings()
        self.setup_tab_math()
        self.setup_tab_persistence()

    def log(self, message):
        self.console.appendPlainText(message)

    def create_nav_layout(self, next_tab_index, next_btn_text="Next Step >"):
        layout = QHBoxLayout()
        layout.addStretch()
        if next_tab_index is not None:
            btn_next = QPushButton(next_btn_text)
            btn_next.clicked.connect(lambda: self.tabs.setCurrentIndex(next_tab_index))
            layout.addWidget(btn_next)
        return layout

    # --- TAB 0: PIPELINE GUIDE ---
    def setup_tab_guide(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        splitter = QSplitter()
        
        list_widget = QListWidget()
        items = [
            "1. Data Preparation",
            "2. Tokenization (BPE)",
            "3. Sequence Generation",
            "4. Token & Position Embeddings",
            "5. Self-Attention Mechanism",
            "6. Multi-Head Attention",
            "7. Feed-Forward Network",
            "8. Residuals & Layer Norm",
            "9. Loss & LR Scheduling",
            "10. Decoding Strategies"
        ]
        list_widget.addItems(items)
        
        text_browser = QTextBrowser()
        text_browser.setOpenExternalLinks(True)
        
        self.guide_content = {
            "1. Data Preparation": """
                <h2 style="color:#03a9f4;">1. Data Preparation</h2>
                <p>LLMs learn by reading vast amounts of text. We use Hugging Face <code>datasets</code> to load text. We filter out empty or very short lines.</p>
            """,
            "2. Tokenization (BPE)": """
                <h2 style="color:#03a9f4;">2. Tokenization (BPE)</h2>
                <p>Neural networks only process numbers. We use <b>Byte-Pair Encoding (BPE)</b>. It merges frequent characters into subwords.</p>
                <p>This allows handling rare words by breaking them into known subwords. We also train special tokens like <code>&lt;bos&gt;</code> and <code>&lt;eos&gt;</code> to teach the model boundaries.</p>
            """,
            "3. Sequence Generation": """
                <h2 style="color:#03a9f4;">3. Sequence Generation</h2>
                <p>LLMs are <b>autoregressive</b>. We chop the token stream into fixed-length sequences (e.g., 24 tokens). For every sequence, we create an Input/Target pair by shifting the tokens by 1.</p>
                <p style="font-family:monospace; background:#333; padding:5px;">Input:  [The, cat, sat, on, the]</p>
                <p style="font-family:monospace; background:#333; padding:5px;">Target: [cat, sat, on, the, mat]</p>
            """,
            "4. Token & Position Embeddings": """
                <h2 style="color:#03a9f4;">4. Token & Position Embeddings</h2>
                <p>Token IDs are mapped to dense vectors. We add a Positional Embedding to the Token Embedding to give the model awareness of word order.</p>
                <p><b>Sinusoidal:</b> Fixed sine/cosine waves (Original 2017 Transformer).</p>
                <p><b>Learned:</b> The model learns the position vectors from scratch (BERT, GPT-2).</p>
                <p><b>Freezing:</b> In transfer learning, embeddings are often frozen so the base vocabulary representations are not destroyed by new gradients.</p>
            """,
            "5. Self-Attention Mechanism": """
                <h2 style="color:#03a9f4;">5. Self-Attention Mechanism</h2>
                <p>For every word, it creates three vectors: Query (Q), Key (K), and Value (V).</p>
                <p>Attention is computed as:</p>
                <p style="font-family:monospace; background:#333; padding:5px;">$Attention(Q, K, V) = softmax(\\frac{Q K^T}{\\sqrt{d_k}}) V$</p>
            """,
            "6. Multi-Head Attention": """
                <h2 style="color:#03a9f4;">6. Multi-Head Attention</h2>
                <p>Transformers run multiple heads in parallel. This allows different heads to focus on different types of relationships simultaneously (e.g., grammar vs semantics).</p>
                <p style="font-family:monospace; background:#333; padding:5px;">$MultiHead = Concat(head_1, ..., head_h) W^O$</p>
            """,
            "7. Feed-Forward Network": """
                <h2 style="color:#03a9f4;">7. Feed-Forward Network (FFN)</h2>
                <p>After attention aggregates context, the FFN processes each position independently. It expands dimensionality, applies a non-linear activation (like <b>GELU</b>), and projects it back down.</p>
            """,
            "8. Residuals & Layer Norm": """
                <h2 style="color:#03a9f4;">8. Residuals & Layer Norm (Pre-LN vs Post-LN)</h2>
                <p><b>Post-LN (Original 2017):</b> $out = LayerNorm(x + Sublayer(x))$. Can be unstable for deep models.</p>
                <p><b>Pre-LN (Modern GPT):</b> $out = x + Sublayer(LayerNorm(x))$. Gradients flow more easily through the residuals, making deep training much more stable.</p>
            """,
            "9. Loss & LR Scheduling": """
                <h2 style="color:#03a9f4;">9. Loss & LR Scheduling</h2>
                <p>We apply <b>Sparse Categorical Cross-Entropy</b> to measure the difference between predicted probabilities and true next tokens.</p>
                <p><b>Warmup + Cosine Decay:</b> The learning rate linearly increases during warmup steps to avoid early gradient explosion, then smoothly decays following a cosine curve. This is the standard schedule for modern LLMs.</p>
                <p><b>Batch Size:</b> Determines how many sequences are processed before updating weights. Smaller batches = noisier gradients but more frequent updates.</p>
            """,
            "10. Decoding Strategies": """
                <h2 style="color:#03a9f4;">10. Decoding Strategies</h2>
                <p><b>Greedy:</b> Picks highest probability token. Causes loops.</p>
                <p><b>Beam Search:</b> Keeps top $B$ sequences. Optimal for factual tasks. Explores a tree of possibilities.</p>
                <p><b>Top-K / Top-P Sampling:</b> Truncates distribution and samples. Optimal for creative writing.</p>
                <p><b>Context Window:</b> The model can only "see" the last $N$ tokens to predict the next one. Older text falls out of the window.</p>
            """
        }
        
        list_widget.currentRowChanged.connect(lambda row: text_browser.setHtml(self.guide_content[items[row]]))
        list_widget.setCurrentRow(0)
        
        splitter.addWidget(list_widget)
        splitter.addWidget(text_browser)
        splitter.setSizes([250, 750])
        layout.addWidget(splitter)
        layout.addLayout(self.create_nav_layout(1))
        self.tabs.addTab(tab, "0. Pipeline Guide")

    # --- TAB 1: CONCEPTS ---
    def setup_tab_concepts(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        browser = QTextBrowser()
        browser.setHtml("""
        <h2 style="color:#03a9f4;">Advanced Academic Guide: Data Science & Interpretability</h2>
        
        <h3 style="color:#03a9f4;">1. Zipf's Law (NLP Fundamental)</h3>
        <p>In any large corpus, the frequency of a word is inversely proportional to its rank in the frequency table. 
        The 2nd most frequent word appears 1/2 as often as the 1st; the 3rd appears 1/3 as often, etc.</p>
        
        <h3 style="color:#03a9f4;">2. Bigram Baselines vs Transformers</h3>
        <p>A Bigram model predicts the next word based <i>solely</i> on the current word (Markov assumption). 
        It lacks context. Comparing Bigram generation to Transformer generation proves the value of Self-Attention.</p>
        
        <h3 style="color:#03a9f4;">3. Top-K Accuracy</h3>
        <p>In language, many words can be syntactically valid next words. Top-1 accuracy is too strict. 
        Top-5 accuracy measures if the correct word is within the model's top 5 highest probability predictions.</p>
        
        <h3 style="color:#03a9f4;">4. Parameter Scaling Dynamics</h3>
        <p>Transformers have specific scaling rules. The Feed-Forward Network (FFN) and Token Embeddings usually contain 80%+ of the parameters. 
        Attention matrices are relatively small. This is why scaling $d_{ff}$ is common.</p>
        
        <h3 style="color:#03a9f4;">5. Transfer Learning & Freezing</h3>
        <p>By freezing the Embedding layer, we lock the vocabulary's vector representations. The model must learn to use existing word meanings to form sentences, rather than inventing new meanings. This reduces trainable parameters drastically.</p>
        """)
        layout.addWidget(browser)
        layout.addLayout(self.create_nav_layout(2))
        self.tabs.addTab(tab, "1. Concepts & Math")

    # --- TAB 2: DATA ---
    def setup_tab_data(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("<h2>Step 1: Load Data</h2>"))
        
        grp_standard = QGroupBox("Standard Datasets")
        std_layout = QVBoxLayout()
        form = QFormLayout()
        self.combo_dataset = QComboBox()
        self.combo_dataset.addItems(["wikitext (Encyclopedic)", "tiny_shakespeare (Poetic - Fast)"])
        self.combo_dataset.setCurrentIndex(1)
        form.addRow("Dataset:", self.combo_dataset)
        std_layout.addLayout(form)
        btn_load = QPushButton("Load Standard Dataset")
        btn_load.clicked.connect(lambda: self.load_data(custom=False))
        std_layout.addWidget(btn_load)
        grp_standard.setLayout(std_layout)
        layout.addWidget(grp_standard)

        grp_custom = QGroupBox("Custom Dataset (Your own text)")
        cust_layout = QVBoxLayout()
        cust_layout.addWidget(QLabel("Upload a raw .txt file to train the LLM on your own data."))
        btn_upload = QPushButton("Upload .txt File")
        btn_upload.clicked.connect(self.upload_custom_data)
        cust_layout.addWidget(btn_upload)
        grp_custom.setLayout(cust_layout)
        layout.addWidget(grp_custom)
        
        self.data_label = QLabel("Dataset not loaded.")
        layout.addWidget(self.data_label)
        self.data_preview = QPlainTextEdit()
        self.data_preview.setReadOnly(True)
        self.data_preview.setMaximumHeight(150)
        layout.addWidget(self.data_preview)
        layout.addLayout(self.create_nav_layout(3))
        self.tabs.addTab(tab, "2. Data")

    def upload_custom_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Text File", "", "Text Files (*.txt)")
        if file_path:
            self.load_data(custom=True, path=file_path)

    def load_data(self, custom=False, path=None):
        ds_name = "wikitext" if "wikitext" in self.combo_dataset.currentText() else "shakespeare"
        self.log("Loading dataset...")
        QApplication.processEvents()
        try:
            num_docs, source = self.llm.load_data(ds_name if not custom else None, path)
            self.data_label.setText(f"Loaded {source}: {num_docs} documents. Total characters: {len(self.llm.raw_text):,}")
            self.data_preview.setPlainText(self.llm.raw_text[:500] + "...")
            self.log(f"{source} loaded successfully!")
        except Exception as e:
            self.log(f"Error: {e}")

    # --- TAB 3: TOKENIZER ---
    def setup_tab_tokenizer(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("<h2>Step 2: BPE Tokenization</h2>"))
        btn_tok = QPushButton("Train BPE Tokenizer")
        btn_tok.clicked.connect(self.build_tokenizer)
        layout.addWidget(btn_tok)
        
        stats_layout = QHBoxLayout()
        self.lbl_vocab = QLabel("Vocab Size: 0")
        self.lbl_chars = QLabel("Total Chars: 0")
        self.lbl_tokens = QLabel("Total Tokens: 0")
        self.lbl_ratio = QLabel("Compression Ratio: 0.0 chars/token")
        for lbl in [self.lbl_vocab, self.lbl_chars, self.lbl_tokens, self.lbl_ratio]:
            lbl.setFrameStyle(QFrame.Panel | QFrame.Sunken)
            stats_layout.addWidget(lbl)
        layout.addLayout(stats_layout)
        
        layout.addWidget(QLabel("<br><b>Tokenizer Inspector:</b>"))
        inspector_layout = QHBoxLayout()
        self.token_input = QLineEdit("The transformer architecture.")
        inspector_layout.addWidget(self.token_input)
        btn_inspect = QPushButton("Inspect Tokens")
        btn_inspect.clicked.connect(self.inspect_tokens)
        inspector_layout.addWidget(btn_inspect)
        layout.addLayout(inspector_layout)
        
        self.token_table = QTableWidget(0, 2)
        self.token_table.setHorizontalHeaderLabels(["Subword Token", "Integer ID"])
        self.token_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.token_table.setMaximumHeight(100)
        layout.addWidget(self.token_table)
        
        layout.addWidget(QLabel("<b>BPE Merge Rules (First 15 learned):</b>"))
        self.merges_list = QListWidget()
        self.merges_list.setMaximumHeight(80)
        layout.addWidget(self.merges_list)
        
        grp_bigram = QGroupBox("Bigram Baseline Generator (Statistical N-gram)")
        bigram_layout = QVBoxLayout()
        bigram_form = QHBoxLayout()
        bigram_form.addWidget(QLabel("Seed:"))
        self.input_bigram_seed = QLineEdit("The king")
        bigram_form.addWidget(self.input_bigram_seed)
        btn_bigram = QPushButton("Generate Bigram Text")
        btn_bigram.clicked.connect(self.generate_bigram)
        bigram_form.addWidget(btn_bigram)
        bigram_layout.addLayout(bigram_form)
        self.output_bigram = QPlainTextEdit()
        self.output_bigram.setReadOnly(True)
        self.output_bigram.setMaximumHeight(80)
        bigram_layout.addWidget(self.output_bigram)
        grp_bigram.setLayout(bigram_layout)
        layout.addWidget(grp_bigram)
        
        plot_splitter = QSplitter(Qt.Vertical)
        
        grp_zipf = QGroupBox("Zipf's Law")
        zipf_layout = QVBoxLayout()
        self.fig_zipf = Figure()
        self.canvas_zipf = FigureCanvas(self.fig_zipf)
        zipf_layout.addWidget(self.canvas_zipf)
        grp_zipf.setLayout(zipf_layout)
        plot_splitter.addWidget(grp_zipf)
        
        grp_doclen = QGroupBox("Document Token Length Distribution")
        doclen_layout = QVBoxLayout()
        self.fig_doclen = Figure()
        self.canvas_doclen = FigureCanvas(self.fig_doclen)
        doclen_layout.addWidget(self.canvas_doclen)
        grp_doclen.setLayout(doclen_layout)
        plot_splitter.addWidget(grp_doclen)
        
        layout.addWidget(plot_splitter)
        layout.addLayout(self.create_nav_layout(4))
        self.tabs.addTab(tab, "3. Tokenizer")

    def build_tokenizer(self):
        if not self.llm.raw_text: return
        self.log("Training BPE Tokenizer...")
        QApplication.processEvents()
        vocab_size, chars, tokens, ratio, merges = self.llm.build_tokenizer()
        self.lbl_vocab.setText(f"Vocab Size: {vocab_size}")
        self.lbl_chars.setText(f"Total Chars: {chars:,}")
        self.lbl_tokens.setText(f"Total Tokens: {tokens:,}")
        self.lbl_ratio.setText(f"Compression Ratio: {ratio:.2f} chars/token")
        self.inspect_tokens()
        
        self.merges_list.clear()
        for m in merges[:15]:
            self.merges_list.addItem(f"{m.replace(' ', ' + ')}")
            
        self.fig_zipf.clear()
        ax = self.fig_zipf.add_subplot(111)
        ranks = np.arange(1, len(self.llm.token_freqs) + 1)
        ax.loglog(ranks, self.llm.token_freqs, marker='.', linestyle='none', color='#03a9f4')
        ax.set_title("Zipf's Law")
        ax.set_xlabel("Token Rank (log)")
        ax.set_ylabel("Token Frequency (log)")
        self.canvas_zipf.draw()
        
        self.fig_doclen.clear()
        ax2 = self.fig_doclen.add_subplot(111)
        ax2.hist(self.llm.doc_lengths, bins=50, color='#4caf50', edgecolor='black')
        ax2.set_title("Document Token Lengths")
        ax2.set_xlabel("Number of Tokens"); ax2.set_ylabel("Number of Documents")
        self.canvas_doclen.draw()

    def inspect_tokens(self):
        if not self.llm.tokenizer: return
        enc = self.llm.tokenizer.encode(self.token_input.text())
        self.token_table.setRowCount(len(enc.tokens))
        for i, (tok, id) in enumerate(zip(enc.tokens, enc.ids)):
            self.token_table.setItem(i, 0, QTableWidgetItem(tok))
            self.token_table.setItem(i, 1, QTableWidgetItem(str(id)))

    def generate_bigram(self):
        if not self.llm.tokenizer: return
        seed = self.input_bigram_seed.text()
        text = self.llm.generate_bigram_text(seed, length=30)
        self.output_bigram.setPlainText(text)

    # --- TAB 4: MODEL ---
    def setup_tab_model(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("<h2>Step 3: Deep Transformer Architecture</h2>"))
        form = QFormLayout()
        
        self.spin_embed = QSpinBox(); self.spin_embed.setRange(32, 256); self.spin_embed.setSingleStep(32); self.spin_embed.setValue(64)
        self.spin_embed.setToolTip("Dimensionality of the token vectors. Larger = more capacity, but slower.")
        form.addRow("Embedding Dim:", self.spin_embed)
        
        self.combo_heads = QComboBox(); self.combo_heads.addItems(["2", "4", "8"])
        self.combo_heads.setToolTip("Number of parallel attention mechanisms.")
        form.addRow("Attention Heads:", self.combo_heads)
        
        self.spin_ff_dim = QSpinBox(); self.spin_ff_dim.setRange(64, 1024); self.spin_ff_dim.setSingleStep(64); self.spin_ff_dim.setValue(256)
        self.spin_ff_dim.setToolTip("Number of neurons in the Feed-Forward Network.")
        form.addRow("Feed-Forward Dim:", self.spin_ff_dim)
        
        self.spin_layers = QSpinBox(); self.spin_layers.setRange(1, 6); self.spin_layers.setValue(2)
        self.spin_layers.setToolTip("Number of stacked Transformer blocks. Deeper = more complex patterns.")
        form.addRow("Num Transformer Layers:", self.spin_layers)
        
        self.combo_pos = QComboBox(); self.combo_pos.addItems(["Learned", "Sinusoidal", "None"])
        self.combo_pos.setToolTip("Learned: Model trains position vectors. Sinusoidal: Fixed sine/cosine waves. None: No position awareness.")
        form.addRow("Positional Encoding:", self.combo_pos)
        
        self.combo_ln = QComboBox(); self.combo_ln.addItems(["Pre-LN (Modern GPT)", "Post-LN (Original 2017)"])
        self.combo_ln.setToolTip("Pre-LN applies LayerNorm before Attention/FFN. It stabilizes deep training gradients.")
        form.addRow("LayerNorm Placement:", self.combo_ln)
        
        self.combo_activation = QComboBox(); self.combo_activation.addItems(["gelu", "relu", "tanh", "elu"]); self.combo_activation.setCurrentText("gelu")
        self.combo_activation.setToolTip("GELU is standard in modern LLMs. It smooths out the harsh cutoff of ReLU.")
        form.addRow("Activation Function:", self.combo_activation)
        
        self.spin_dropout = QDoubleSpinBox(); self.spin_dropout.setRange(0.0, 0.5); self.spin_dropout.setSingleStep(0.1); self.spin_dropout.setValue(0.1)
        self.spin_dropout.setToolTip("Drops out random neurons during training to prevent overfitting.")
        form.addRow("Dropout Rate:", self.spin_dropout)
        
        self.combo_opt = QComboBox(); self.combo_opt.addItems(["AdamW", "Adam"])
        self.combo_opt.setToolTip("AdamW decouples weight decay from the gradient update.")
        form.addRow("Optimizer:", self.combo_opt)
        
        self.spin_wd = QDoubleSpinBox(); self.spin_wd.setRange(0.0, 0.1); self.spin_wd.setSingleStep(0.001); self.spin_wd.setValue(0.01)
        self.spin_wd.setToolTip("L2 regularization. Penalizes large weights to improve generalization.")
        form.addRow("Weight Decay:", self.spin_wd)
        
        self.spin_clip = QDoubleSpinBox(); self.spin_clip.setRange(0.0, 10.0); self.spin_clip.setSingleStep(0.1); self.spin_clip.setValue(1.0)
        self.spin_clip.setToolTip("Caps the L2 norm of gradients. Prevents 'exploding gradients'.")
        form.addRow("Gradient Clipping Norm:", self.spin_clip)
        
        self.spin_batch = QSpinBox(); self.spin_batch.setRange(4, 64); self.spin_batch.setSingleStep(4); self.spin_batch.setValue(32)
        self.spin_batch.setToolTip("Number of sequences processed before updating weights. Smaller = noisier gradients.")
        form.addRow("Batch Size:", self.spin_batch)
        
        self.spin_epochs_model = QSpinBox(); self.spin_epochs_model.setRange(1, 50); self.spin_epochs_model.setValue(15)
        self.spin_epochs_model.setToolTip("Total epochs. Needed to calculate the Cosine Decay schedule.")
        form.addRow("Epochs (for LR Schedule):", self.spin_epochs_model)
        
        self.spin_warmup = QSpinBox(); self.spin_warmup.setRange(0, 200); self.spin_warmup.setValue(20)
        self.spin_warmup.setToolTip("Number of steps to linearly increase LR from 0 to base_lr.")
        form.addRow("LR Warmup Steps:", self.spin_warmup)
        
        self.chk_freeze = QCheckBox("Freeze Token Embeddings (Transfer Learning)")
        self.chk_freeze.setToolTip("Locks the vocabulary vectors. Model only updates Transformer blocks.")
        form.addRow("", self.chk_freeze)
        
        layout.addLayout(form)
        
        btn_model = QPushButton("Build Neural Network")
        btn_model.clicked.connect(self.build_model)
        layout.addWidget(btn_model)
        self.model_label = QLabel("Model not built.")
        layout.addWidget(self.model_label)
        
        viz_splitter = QSplitter()
        self.model_summary = QPlainTextEdit(); self.model_summary.setReadOnly(True)
        viz_splitter.addWidget(self.model_summary)
        
        self.fig_params = Figure()
        self.canvas_params = FigureCanvas(self.fig_params)
        viz_splitter.addWidget(self.canvas_params)
        
        layout.addWidget(viz_splitter)
        layout.addLayout(self.create_nav_layout(5))
        self.tabs.addTab(tab, "4. Model")

    def build_model(self):
        if self.llm.vocab_size == 0: return
        self.log("Building model...")
        use_pre_ln = "Pre-LN" in self.combo_ln.currentText()
        pos_type = self.combo_pos.currentText().lower()
        params, summary, param_breakdown, total_params, trainable_params = self.llm.build_model(
            self.spin_embed.value(), int(self.combo_heads.currentText()),
            self.spin_ff_dim.value(), self.spin_layers.value(), self.combo_activation.currentText(),
            self.combo_opt.currentText(), self.spin_clip.value(), use_pre_ln, pos_type,
            self.spin_dropout.value(), self.spin_wd.value(),
            self.spin_epochs_model.value(), self.spin_warmup.value(),
            self.spin_batch.value(), self.chk_freeze.isChecked()
        )
        size_mb = (params * 4) / (1024 * 1024)
        self.model_label.setText(f"Model built.\nTotal Params: {params:,} (Trainable: {trainable_params:,}) | Size: {size_mb:.2f} MB")
        self.model_summary.setPlainText(summary)
        
        self.fig_params.clear()
        ax = self.fig_params.add_subplot(111)
        labels = list(param_breakdown.keys())
        sizes = list(param_breakdown.values())
        colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'color':'w', 'fontsize':8})
        ax.axis('equal')
        ax.set_title(f"Parameter Distribution (Total: {total_params:,})")
        self.canvas_params.draw()

    # --- TAB 5: TRAINING ---
    def setup_tab_train(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("<h2>Step 4: Training</h2>"))
        form = QFormLayout()
        self.spin_epochs = QSpinBox(); self.spin_epochs.setRange(1, 50); self.spin_epochs.setValue(15)
        form.addRow("Epochs:", self.spin_epochs)
        layout.addLayout(form)
        
        btn_layout = QHBoxLayout()
        self.btn_train = QPushButton("Start Training")
        self.btn_train.clicked.connect(self.start_training)
        btn_layout.addWidget(self.btn_train)
        self.btn_eval = QPushButton("Evaluate Full Val Set")
        self.btn_eval.clicked.connect(self.evaluate_full_val)
        btn_layout.addWidget(self.btn_eval)
        layout.addLayout(btn_layout)
        
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        
        metric_frame = QFrame()
        metric_frame.setObjectName("MetricFrame")
        metric_layout = QHBoxLayout(metric_frame)
        self.lbl_m_loss = QLabel("Train Loss: -")
        self.lbl_m_val = QLabel("Val Loss: -")
        self.lbl_m_ppl = QLabel("Perplexity: -")
        self.lbl_m_grad = QLabel("Grad Norm: -")
        self.lbl_m_lr = QLabel("Learning Rate: -")
        self.lbl_m_top1 = QLabel("Top-1 Acc: -")
        self.lbl_m_top5 = QLabel("Top-5 Acc: -")
        for lbl in [self.lbl_m_loss, self.lbl_m_val, self.lbl_m_ppl, self.lbl_m_grad, self.lbl_m_lr, self.lbl_m_top1, self.lbl_m_top5]:
            metric_layout.addWidget(lbl)
        layout.addWidget(metric_frame)
        
        splitter = QSplitter(Qt.Vertical)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        splitter.addWidget(self.canvas)
        
        self.fig_grads = Figure()
        self.canvas_grads = FigureCanvas(self.fig_grads)
        self.ax_grads = self.fig_grads.add_subplot(111)
        self.ax_grads.set_title("Embedding Gradient Distribution")
        splitter.addWidget(self.canvas_grads)
        
        self.fig_layer_norms = Figure()
        self.canvas_layer_norms = FigureCanvas(self.fig_layer_norms)
        self.ax_layer_norms = self.fig_layer_norms.add_subplot(111)
        self.ax_layer_norms.set_title("Per-Layer Gradient Norm (Vanishing Gradients)")
        splitter.addWidget(self.canvas_layer_norms)
        
        layout.addWidget(splitter)
        
        self.ax1 = self.figure.add_subplot(411)
        self.ax2 = self.figure.add_subplot(412)
        self.ax3 = self.figure.add_subplot(413)
        self.ax4 = self.figure.add_subplot(414)
        self.ax1.set_title("Loss (Train vs Val)"); self.ax2.set_title("Top-K Validation Accuracy")
        self.ax3.set_title("Gradient Norm"); self.ax4.set_title("Learning Rate (Warmup + Cosine)")
        self.figure.tight_layout()
        self.losses, self.val_losses, self.grads, self.lrs = [], [], [], []
        self.top1s, self.top3s, self.top5s = [], [], []
        layout.addLayout(self.create_nav_layout(6))
        self.tabs.addTab(tab, "5. Train")

    def evaluate_full_val(self):
        if not self.llm.model: return
        self.log("Evaluating full validation set...")
        QApplication.processEvents()
        total_loss = 0.0
        total_samples = 0
        top1, top5, total = 0, 0, 0
        
        for x_val, y_val in self.llm.val_dataset:
            preds = self.llm.model(x_val, training=False)
            loss = self.llm.model.compiled_loss(y_val, preds)
            total_loss += float(loss) * x_val.shape[0]
            total_samples += x_val.shape[0]
            
            top_k_preds = tf.math.top_k(preds, k=5).indices.numpy()
            y_val_np = y_val.numpy()
            for i in range(y_val_np.shape[0]):
                for j in range(y_val_np.shape[1]):
                    true_id = y_val_np[i, j]
                    if true_id == 0: continue
                    top_k = top_k_preds[i, j]
                    if true_id in top_k[:1]: top1 += 1
                    if true_id in top_k[:5]: top5 += 1
                    total += 1
                    
        avg_loss = total_loss / total_samples
        ppl = float(np.exp(min(avg_loss, 20)))
        t1 = top1 / total if total > 0 else 0
        t5 = top5 / total if total > 0 else 0
        
        QMessageBox.information(self, "Corpus Evaluation", f"Validation Loss: {avg_loss:.4f}\nPerplexity: {ppl:.2f}\nTop-1 Acc: {t1:.2%}\nTop-5 Acc: {t5:.2%}")
        self.log(f"Eval complete. Loss: {avg_loss:.4f}, PPL: {ppl:.2f}, Top-1: {t1:.2%}, Top-5: {t5:.2%}")

    def start_training(self):
        if not self.llm.model: return
        self.btn_train.setEnabled(False); self.progress.setValue(0)
        self.losses, self.val_losses, self.grads, self.lrs = [], [], [], []
        self.top1s, self.top3s, self.top5s = [], [], []
        self.ax1.clear(); self.ax2.clear(); self.ax3.clear(); self.ax4.clear()
        self.ax1.set_title("Loss (Train vs Val)"); self.ax2.set_title("Top-K Validation Accuracy")
        self.ax3.set_title("Gradient Norm"); self.ax4.set_title("Learning Rate (Warmup + Cosine)")
        self.canvas.draw()
        self.training_thread = TrainingThread(self.llm, self.spin_epochs.value())
        self.training_thread.update_log.connect(self.log)
        self.training_thread.update_progress.connect(self.on_epoch_update)
        self.training_thread.training_done.connect(lambda: self.btn_train.setEnabled(True))
        self.training_thread.start()

    def on_epoch_update(self, progress, m):
        self.progress.setValue(progress)
        self.losses.append(m['loss']); self.val_losses.append(m['val_loss']); self.grads.append(m['grad_norm']); self.lrs.append(m['lr'])
        self.top1s.append(m['top1']); self.top3s.append(m['top3']); self.top5s.append(m['top5'])
        
        self.ax1.clear(); self.ax2.clear(); self.ax3.clear(); self.ax4.clear()
        self.ax1.plot(self.losses, label="Train", color='#03a9f4')
        self.ax1.plot(self.val_losses, label="Val", color='#e91e63')
        self.ax1.legend()
        
        self.ax2.plot(self.top1s, label="Top-1", color='#03a9f4')
        self.ax2.plot(self.top3s, label="Top-3", color='#4caf50')
        self.ax2.plot(self.top5s, label="Top-5", color='#ff9800')
        self.ax2.legend()
        
        self.ax3.plot(self.grads, color='#4caf50')
        self.ax4.plot(self.lrs, color='#ff9800')
        
        self.ax1.set_title("Loss (Train vs Val)"); self.ax2.set_title("Top-K Validation Accuracy")
        self.ax3.set_title("Gradient Norm"); self.ax4.set_title("Learning Rate (Warmup + Cosine)")
        self.figure.tight_layout()
        self.canvas.draw()
        
        if m['emb_grads']:
            self.ax_grads.clear()
            sample = np.random.choice(m['emb_grads'], size=min(len(m['emb_grads']), 5000), replace=False)
            self.ax_grads.hist(sample, bins=50, color='#9c27b0', edgecolor='black')
            self.ax_grads.set_title("Embedding Gradient Distribution (Sampled)")
            self.ax_grads.set_xlabel("Gradient Value"); self.ax_grads.set_ylabel("Frequency")
            self.fig_grads.tight_layout()
            self.canvas_grads.draw()
            
        if m['layer_norms']:
            self.ax_layer_norms.clear()
            names = [ln[0] for ln in m['layer_norms']]
            values = [ln[1] for ln in m['layer_norms']]
            colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
            self.ax_layer_norms.bar(names, values, color=colors)
            self.ax_layer_norms.set_title("Gradient Norm per Layer Block")
            self.ax_layer_norms.set_xlabel("Layer"); self.ax_layer_norms.set_ylabel("Gradient L2 Norm")
            self.ax_layer_norms.tick_params(axis='x', rotation=45)
            self.fig_layer_norms.tight_layout()
            self.canvas_layer_norms.draw()
        
        self.lbl_m_loss.setText(f"Train Loss: <b>{m['loss']:.4f}</b>")
        self.lbl_m_val.setText(f"Val Loss: <b>{m['val_loss']:.4f}</b>")
        self.lbl_m_ppl.setText(f"Perplexity: <b>{m['ppl']:.2f}</b>")
        self.lbl_m_grad.setText(f"Grad Norm: <b>{m['grad_norm']:.4f}</b>")
        self.lbl_m_lr.setText(f"Learning Rate: <b>{m['lr']:.6f}</b>")
        self.lbl_m_top1.setText(f"Top-1 Acc: <b>{m['top1']:.2%}</b>")
        self.lbl_m_top5.setText(f"Top-5 Acc: <b>{m['top5']:.2%}</b>")

    # --- TAB 6: GENERATE & INTERPRET ---
    def setup_tab_generate(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("<h2>Step 5: Generation & Interpretability</h2>"))
        gen_form = QHBoxLayout()
        self.input_seed = QLineEdit("The king and the queen")
        gen_form.addWidget(QLabel("Seed:"))
        gen_form.addWidget(self.input_seed)
        layout.addLayout(gen_form)
        
        form = QFormLayout()
        self.combo_decode = QComboBox(); self.combo_decode.addItems(["Sampling", "Greedy", "Beam Search"])
        self.combo_decode.setToolTip("Sampling: Stochastic. Greedy: Deterministic. Beam Search: Keeps top B paths.")
        form.addRow("Decoding Strategy:", self.combo_decode)
        self.spin_temp = QDoubleSpinBox(); self.spin_temp.setRange(0.1, 2.0); self.spin_temp.setValue(0.7)
        self.spin_temp.setToolTip("Scales logits before Softmax. <1.0 = confident. >1.0 = random.")
        form.addRow("Temperature:", self.spin_temp)
        self.spin_rep_pen = QDoubleSpinBox(); self.spin_rep_pen.setRange(1.0, 2.0); self.spin_rep_pen.setSingleStep(0.1); self.spin_rep_pen.setValue(1.2)
        self.spin_rep_pen.setToolTip("Penalizes tokens that have already appeared. Prevents loops.")
        form.addRow("Repetition Penalty:", self.spin_rep_pen)
        self.spin_top_k = QSpinBox(); self.spin_top_k.setRange(0, 100); self.spin_top_k.setValue(40)
        self.spin_top_k.setToolTip("Truncates vocabulary to top K tokens before sampling.")
        form.addRow("Top-K (0=off):", self.spin_top_k)
        self.spin_top_p = QDoubleSpinBox(); self.spin_top_p.setRange(0.0, 1.0); self.spin_top_p.setSingleStep(0.05); self.spin_top_p.setValue(0.90)
        self.spin_top_p.setToolTip("Keeps smallest set of tokens whose cumulative probability exceeds P.")
        form.addRow("Top-P (1.0=off):", self.spin_top_p)
        self.spin_beam = QSpinBox(); self.spin_beam.setRange(2, 5); self.spin_beam.setValue(3)
        form.addRow("Beam Width:", self.spin_beam)
        self.spin_gen_len = QSpinBox(); self.spin_gen_len.setRange(10, 200); self.spin_gen_len.setValue(30)
        form.addRow("Max Generation Length:", self.spin_gen_len)
        layout.addLayout(form)
        
        btn_layout = QHBoxLayout()
        self.btn_generate = QPushButton("Start Streaming Generation")
        self.btn_generate.clicked.connect(self.start_generation)
        btn_layout.addWidget(self.btn_generate)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("StopBtn")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_generation)
        btn_layout.addWidget(self.btn_stop)
        layout.addLayout(btn_layout)
        
        metric_frame = QFrame()
        metric_frame.setObjectName("MetricFrame")
        metric_layout = QHBoxLayout(metric_frame)
        self.lbl_tps = QLabel("Tokens/sec: 0.0")
        self.lbl_entropy = QLabel("Output Entropy: 0.0")
        metric_layout.addWidget(self.lbl_tps)
        metric_layout.addWidget(self.lbl_entropy)
        layout.addWidget(metric_frame)
        
        nll_layout = QHBoxLayout()
        nll_layout.addWidget(QLabel("Eval Sequence:"))
        self.input_eval = QLineEdit("The quick brown fox")
        nll_layout.addWidget(self.input_eval)
        btn_eval = QPushButton("Calc Sequence NLL & Heatmap")
        btn_eval.clicked.connect(self.calculate_nll)
        nll_layout.addWidget(btn_eval)
        self.lbl_nll = QLabel("NLL: - | PPL: -")
        nll_layout.addWidget(self.lbl_nll)
        layout.addLayout(nll_layout)
        
        inner_tabs = QTabWidget()
        
        text_tab = QWidget()
        text_layout = QVBoxLayout(text_tab)
        splitter = QSplitter()
        
        # Changed to QTextEdit to support HTML highlighting
        self.output_text = QTextEdit(); self.output_text.setReadOnly(True)
        splitter.addWidget(self.output_text)
        
        prob_splitter = QSplitter(Qt.Vertical)
        self.fig_prob = Figure(); self.canvas_prob = FigureCanvas(self.fig_prob)
        self.ax_prob = self.fig_prob.add_subplot(111)
        self.ax_prob.set_title("Token Probability Distribution (Top 50)")
        
        self.fig_traj = Figure(); self.canvas_traj = FigureCanvas(self.fig_traj)
        self.ax_traj = self.fig_traj.add_subplot(111)
        self.ax_traj.set_title("Chosen Token Probability Trajectory")
        self.ax_traj.set_xlabel("Step"); self.ax_traj.set_ylabel("Probability")
        
        self.fig_margin = Figure(); self.canvas_margin = FigureCanvas(self.fig_margin)
        self.ax_margin = self.fig_margin.add_subplot(111)
        self.ax_margin.set_title("Logit Margin (Top1 - Top2)")
        self.ax_margin.set_xlabel("Step"); self.ax_margin.set_ylabel("Margin")
        
        prob_splitter.addWidget(self.canvas_prob)
        prob_splitter.addWidget(self.canvas_traj)
        prob_splitter.addWidget(self.canvas_margin)
        splitter.addWidget(prob_splitter)
        
        text_layout.addWidget(splitter)
        inner_tabs.addTab(text_tab, "Text & Probabilities")
        
        att_tab = QWidget()
        att_layout = QVBoxLayout(att_tab)
        att_controls = QHBoxLayout()
        att_controls.addWidget(QLabel("Visualize Attention Layer:"))
        self.combo_att_layer = QComboBox()
        att_controls.addWidget(self.combo_att_layer)
        att_controls.addStretch()
        att_layout.addLayout(att_controls)
        att_layout.addWidget(QLabel("<b>Multi-Head Attention Heatmap (with Entropy):</b>"))
        self.fig_att = Figure(); self.canvas_att = FigureCanvas(self.fig_att)
        att_layout.addWidget(self.canvas_att)
        inner_tabs.addTab(att_tab, "Attention Heatmap")
        
        beam_tab = QWidget()
        beam_layout = QVBoxLayout(beam_tab)
        beam_layout.addWidget(QLabel("<b>Beam Search Tree Explorer:</b>"))
        self.fig_beam = Figure(); self.canvas_beam = FigureCanvas(self.fig_beam)
        beam_layout.addWidget(self.canvas_beam)
        inner_tabs.addTab(beam_tab, "Beam Search Tree")
        
        bench_tab = QWidget()
        bench_layout = QVBoxLayout(bench_tab)
        bench_layout.addWidget(QLabel("<b>Attention Inference Benchmark ($O(N^2)$):</b>"))
        btn_bench = QPushButton("Run Inference Complexity Benchmark")
        btn_bench.clicked.connect(self.run_inference_benchmark)
        bench_layout.addWidget(btn_bench)
        self.fig_bench = Figure(); self.canvas_bench = FigureCanvas(self.fig_bench)
        bench_layout.addWidget(self.canvas_bench)
        inner_tabs.addTab(bench_tab, "Complexity Benchmark")
        
        logits_tab = QWidget()
        logits_layout = QVBoxLayout(logits_tab)
        logits_layout.addWidget(QLabel("<b>Sequence Logits Heatmap (Top 15 Tokens per Position):</b>"))
        self.fig_logits = Figure(); self.canvas_logits = FigureCanvas(self.fig_logits)
        logits_layout.addWidget(self.canvas_logits)
        inner_tabs.addTab(logits_tab, "Sequence Logits")
        
        nll_tab = QWidget()
        nll_layout_tab = QVBoxLayout(nll_tab)
        nll_layout_tab.addWidget(QLabel("<b>Token-Level Loss Heatmap (Per-token NLL):</b>"))
        self.fig_nll = Figure(); self.canvas_nll = FigureCanvas(self.fig_nll)
        nll_layout_tab.addWidget(self.canvas_nll)
        inner_tabs.addTab(nll_tab, "Token Loss Heatmap")
        
        layout.addWidget(inner_tabs)
        self.tabs.addTab(tab, "6. Generate & Interpret")

    def run_inference_benchmark(self):
        if not self.llm.model: return
        self.log("Running inference benchmark...")
        QApplication.processEvents()
        lengths = range(1, self.llm.seq_length + 1)
        times = []
        for l in lengths:
            dummy_ids = [1] * l
            padding = [0] * (self.llm.seq_length - l)
            token_ids = padding + dummy_ids
            start = time.time()
            for _ in range(10):
                self.llm.model.predict(np.array([token_ids]), verbose=0)
            times.append((time.time() - start) / 10)
        self.fig_bench.clear()
        ax = self.fig_bench.add_subplot(111)
        ax.plot(list(lengths), times, marker='o', color='#e91e63')
        ax.set_title("Inference Time vs. Context Length")
        ax.set_xlabel("Sequence Length (N)")
        ax.set_ylabel("Time (s)")
        self.canvas_bench.draw()

    def calculate_nll(self):
        if not self.llm.model: return
        text = self.input_eval.text().strip()
        if not text: return
        enc = self.llm.tokenizer.encode(text)
        ids = enc.ids
        words = enc.tokens
        if len(ids) > self.llm.seq_length: ids = ids[-self.llm.seq_length:]; words = words[-self.llm.seq_length:]
        if len(ids) < self.llm.seq_length:
            padding = [0] * (self.llm.seq_length - len(ids))
            ids = padding + ids; words = ["[PAD]"]*len(padding) + words
        predictions = self.llm.model.predict(np.array([ids]), verbose=0)
        log_probs = tf.nn.log_softmax(predictions[0], axis=-1).numpy()
        
        total_nll = 0.0; count = 0
        token_losses = []
        for i in range(1, len(ids)):
            loss = -log_probs[i-1, ids[i]]
            token_losses.append(float(loss))
            total_nll += loss
            count += 1
        avg_nll = total_nll / count if count > 0 else 0
        ppl = float(np.exp(min(avg_nll, 20)))
        self.lbl_nll.setText(f"NLL: <b>{avg_nll:.4f}</b> | PPL: <b>{ppl:.2f}</b>")
        
        self.fig_nll.clear()
        ax = self.fig_nll.add_subplot(111)
        valid_words = words[1:len(token_losses)+1]
        ax.bar(valid_words, token_losses, color='#e91e63')
        ax.set_title("Negative Log-Likelihood per Token")
        ax.set_xlabel("Token"); ax.set_ylabel("NLL Loss")
        ax.tick_params(axis='x', rotation=45)
        self.fig_nll.tight_layout()
        self.canvas_nll.draw()

    def start_generation(self):
        if not self.llm.model: return
        if self.generation_thread and self.generation_thread.isRunning(): return
        self.btn_generate.setEnabled(False); self.btn_stop.setEnabled(True)
        self.output_text.clear()
        self.ax_traj.clear(); self.ax_margin.clear()
        self.ax_traj.set_title("Chosen Token Probability Trajectory")
        self.ax_margin.set_title("Logit Margin (Top1 - Top2)")
        self.canvas_traj.draw(); self.canvas_margin.draw()
        
        self.combo_att_layer.clear()
        for i in range(len(self.llm.transformer_blocks)):
            self.combo_att_layer.addItem(f"Layer {i+1}")
        
        self.generation_thread = GenerationThread(
            self.llm, self.input_seed.text(), self.spin_gen_len.value(), 
            self.spin_temp.value(), self.spin_top_k.value(), self.spin_top_p.value(),
            self.combo_decode.currentText(), self.spin_rep_pen.value(), self.spin_beam.value()
        )
        self.generation_thread.update_text.connect(self.on_update_text)
        self.generation_thread.update_probs.connect(self.on_update_probs)
        self.generation_thread.update_metrics.connect(self.on_update_gen_metrics)
        self.generation_thread.update_traj.connect(self.on_update_traj)
        self.generation_thread.update_margin.connect(self.on_update_margin)
        self.generation_thread.update_beam_tree.connect(self.on_update_beam_tree)
        self.generation_thread.generation_done.connect(self.on_generation_done)
        self.generation_thread.start()

    def stop_generation(self):
        if self.generation_thread:
            self.generation_thread.stop()
            self.log("Generation stopped by user.")

    def on_update_text(self, text, is_final):
        # Highlight the context window (last seq_length words)
        words = text.split()
        if len(words) > self.llm.seq_length and not is_final:
            context = words[-self.llm.seq_length:]
            prefix = " ".join(words[:-self.llm.seq_length])
            html = prefix + " <span style='background-color:#3c3c3c; border-radius:3px; padding:2px;'>" + " ".join(context) + "</span>"
            self.output_text.setHtml(html)
        else:
            self.output_text.setPlainText(text)

    def on_update_probs(self, tokens, top_10_probs, top_50_probs):
        self.ax_prob.clear()
        self.ax_prob.plot(top_50_probs, marker='o', color='#03a9f4', linewidth=2)
        self.ax_prob.set_title("Token Probability Distribution (Top 50)")
        self.ax_prob.set_xlabel("Token Rank (0 to 50)"); self.ax_prob.set_ylabel("Probability")
        self.ax_prob.set_ylim(0, max(top_50_probs)*1.1 if top_50_probs else 1)
        self.canvas_prob.draw()

    def on_update_traj(self, step, prob):
        self.ax_traj.plot(step, prob, 'o-', color='#e91e63')
        self.ax_traj.set_xlim(0, self.spin_gen_len.value()); self.ax_traj.set_ylim(0, 1)
        self.canvas_traj.draw()

    def on_update_margin(self, step, margin):
        self.ax_margin.plot(step, margin, 'o-', color='#4caf50')
        self.ax_margin.set_xlim(0, self.spin_gen_len.value())
        self.canvas_margin.draw()

    def on_update_gen_metrics(self, tps, entropy):
        self.lbl_tps.setText(f"Tokens/sec: <b>{tps:.2f}</b>")
        self.lbl_entropy.setText(f"Output Entropy: <b>{entropy:.4f}</b>")

    def on_update_beam_tree(self, edges):
        self.fig_beam.clear()
        ax = self.fig_beam.add_subplot(111)
        G = nx.DiGraph()
        
        for parent, child, score in edges:
            G.add_edge(parent, child, weight=score)
            
        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
        
        nx.draw(G, pos, ax=ax, with_labels=False, node_size=50, node_color='#03a9f4', arrows=True)
        labels = {n: n[-15:] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=6, font_color='white', ax=ax)
        
        ax.set_title("Beam Search Expansion Tree")
        self.canvas_beam.draw()

    def on_generation_done(self, final_token_ids, final_logits):
        self.btn_generate.setEnabled(True); self.btn_stop.setEnabled(False)
        self.log("Generation complete. Rendering attention and logits...")
        self.plot_attention_heads(final_token_ids)
        self.plot_sequence_logits(final_token_ids, final_logits)

    def plot_attention_heads(self, token_ids):
        if not self.llm.transformer_blocks: return
        layer_idx = int(self.combo_att_layer.currentText().split(" ")[1]) - 1
        selected_block = self.llm.transformer_blocks[layer_idx]
        attn_weights = selected_block.last_attn_scores[0] 
        num_heads = attn_weights.shape[0]
        actual_len = np.count_nonzero(token_ids)
        words = self.llm.tokenizer.decode(token_ids[-actual_len:]).split()
        offset = self.llm.seq_length - actual_len
        attn_weights = attn_weights[:, offset:, offset:]
        rows = int(np.ceil(num_heads / 2)); cols = 2
        self.fig_att.clear()
        for i in range(num_heads):
            ax = self.fig_att.add_subplot(rows, cols, i+1)
            cax = ax.imshow(attn_weights[i], cmap='viridis')
            ax.set_xticks(np.arange(len(words))); ax.set_yticks(np.arange(len(words)))
            ax.set_xticklabels(words, rotation=90, fontsize=8); ax.set_yticklabels(words, fontsize=8)
            if len(words) <= 10:
                for r in range(len(words)):
                    for c in range(len(words)):
                        val = attn_weights[i, r, c]
                        color = "white" if val < 0.5 else "black"
                        ax.text(c, r, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)
            head_attn = attn_weights[i] + 1e-10
            entropy = -np.sum(head_attn * np.log(head_attn))
            ax.set_title(f"Head {i+1} (Entropy: {entropy:.2f})")
            self.fig_att.colorbar(cax, ax=ax)
        self.fig_att.tight_layout()
        self.canvas_att.draw()

    def plot_sequence_logits(self, token_ids, logits_matrix):
        actual_len = np.count_nonzero(token_ids)
        words = self.llm.tokenizer.decode(token_ids[-actual_len:]).split()
        valid_logits = logits_matrix[-actual_len:]
        top_indices = np.argsort(np.max(valid_logits, axis=0))[-15:][::-1]
        labels = [self.llm.tokenizer.decode([int(i)]) for i in top_indices]
        data = valid_logits[:, top_indices].T
        
        self.fig_logits.clear()
        ax = self.fig_logits.add_subplot(111)
        cax = ax.imshow(data, cmap='magma', aspect='auto')
        ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels)
        ax.set_xticks(np.arange(len(words))); ax.set_xticklabels(words, rotation=90)
        ax.set_title("Raw Logits for Top 15 Tokens Across Sequence")
        ax.set_xlabel("Sequence Position"); ax.set_ylabel("Top Tokens")
        self.fig_logits.colorbar(cax, ax=ax)
        self.fig_logits.tight_layout()
        self.canvas_logits.draw()

    # --- TAB 7: EMBEDDINGS LAB ---
    def setup_tab_embeddings(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("<h2>Step 6: Embedding Space Laboratory</h2>"))
        
        grp_math = QGroupBox("Vector Arithmetic (Analogies)")
        math_layout = QVBoxLayout()
        math_form = QFormLayout()
        self.input_a = QLineEdit("king"); self.input_b = QLineEdit("man"); self.input_c = QLineEdit("woman")
        math_form.addRow("Vector A:", self.input_a); math_form.addRow("Minus B:", self.input_b); math_form.addRow("Plus C:", self.input_c)
        math_layout.addLayout(math_form)
        btn_math = QPushButton("Compute: A - B + C"); btn_math.clicked.connect(self.compute_analogy)
        math_layout.addWidget(btn_math)
        self.table_math = QTableWidget(0, 2); self.table_math.setHorizontalHeaderLabels(["Closest Word", "Cosine Sim"])
        self.table_math.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); self.table_math.setMaximumHeight(120)
        math_layout.addWidget(self.table_math)
        grp_math.setLayout(math_layout); layout.addWidget(grp_math)

        grp_ctx = QGroupBox("Static vs Contextual Embeddings")
        ctx_layout = QVBoxLayout()
        ctx_form = QFormLayout()
        self.input_word = QLineEdit("bank")
        self.input_s1 = QLineEdit("The river bank was muddy.")
        self.input_s2 = QLineEdit("He went to the bank for a loan.")
        ctx_form.addRow("Target Word:", self.input_word); ctx_form.addRow("Sentence 1:", self.input_s1); ctx_form.addRow("Sentence 2:", self.input_s2)
        
        ctx_layer_layout = QHBoxLayout()
        ctx_layer_layout.addWidget(QLabel("Extract from Layer:"))
        self.combo_ctx_layer = QComboBox()
        ctx_layer_layout.addWidget(self.combo_ctx_layer)
        ctx_form.addRow(ctx_layer_layout)
        
        ctx_layout.addLayout(ctx_form)
        btn_ctx = QPushButton("Compare Contexts (Cosine Similarity)"); btn_ctx.clicked.connect(self.compute_contextual)
        ctx_layout.addWidget(btn_ctx)
        self.label_ctx = QLabel("Similarity will appear here.")
        ctx_layout.addWidget(self.label_ctx)
        grp_ctx.setLayout(ctx_layout); layout.addWidget(grp_ctx)

        grp_sim = QGroupBox("Contextual Similarity Matrix Heatmap")
        sim_layout = QVBoxLayout()
        sim_form = QHBoxLayout()
        sim_form.addWidget(QLabel("Sentence:"))
        self.input_sim_sent = QLineEdit("The quick brown fox jumps over the lazy dog")
        sim_form.addWidget(self.input_sim_sent)
        
        sim_layer_layout = QHBoxLayout()
        sim_layer_layout.addWidget(QLabel("Layer:"))
        self.combo_sim_layer = QComboBox()
        sim_layer_layout.addWidget(self.combo_sim_layer)
        sim_form.addWidget(sim_layer_layout)
        
        btn_sim = QPushButton("Plot Matrix"); btn_sim.clicked.connect(self.plot_contextual_matrix)
        sim_form.addWidget(btn_sim)
        sim_layout.addLayout(sim_form)
        self.fig_sim = Figure(); self.canvas_sim = FigureCanvas(self.fig_sim)
        sim_layout.addWidget(self.canvas_sim)
        grp_sim.setLayout(sim_layout); layout.addWidget(grp_sim)

        grp_viz = QGroupBox("Vector Space & Weight Visualization")
        viz_layout = QVBoxLayout()
        viz_controls = QHBoxLayout(); viz_controls.addWidget(QLabel("Algorithm:"))
        self.radio_pca = QRadioButton("PCA"); self.radio_tsne = QRadioButton("t-SNE"); self.radio_pca.setChecked(True)
        viz_controls.addWidget(self.radio_pca); viz_controls.addWidget(self.radio_tsne); viz_controls.addStretch()
        btn_plot_2d = QPushButton("Project to 2D"); btn_plot_2d.clicked.connect(self.plot_embeddings_2d); viz_controls.addWidget(btn_plot_2d)
        btn_plot_norms = QPushButton("Plot Vector Norms"); btn_plot_norms.clicked.connect(self.plot_norms); viz_controls.addWidget(btn_plot_norms)
        btn_plot_weights = QPushButton("Plot Model Weights"); btn_plot_weights.clicked.connect(self.plot_model_weights); viz_controls.addWidget(btn_plot_weights)
        btn_plot_freq_norm = QPushButton("Plot Freq vs Norm"); btn_plot_freq_norm.clicked.connect(self.plot_freq_vs_norm); viz_controls.addWidget(btn_plot_freq_norm)
        viz_layout.addLayout(viz_controls)
        self.fig_emb = Figure(); self.canvas_emb = FigureCanvas(self.fig_emb); viz_layout.addWidget(self.canvas_emb)
        grp_viz.setLayout(viz_layout); layout.addWidget(grp_viz)
        self.tabs.addTab(tab, "7. Embeddings Lab")

    def plot_contextual_matrix(self):
        if not self.llm.model: return
        sentence = self.input_sim_sent.text().strip()
        enc = self.llm.tokenizer.encode(sentence)
        ids = enc.ids; words = enc.tokens
        if len(ids) > self.llm.seq_length:
            ids = ids[-self.llm.seq_length:]; words = words[-self.llm.seq_length:]
        if len(ids) < self.llm.seq_length:
            padding = [0] * (self.llm.seq_length - len(ids))
            ids = padding + ids; words = ["[PAD]"]*len(padding) + words
        
        layer_idx = int(self.combo_sim_layer.currentText().split(" ")[1])
        extractor = self.llm.context_extractors[layer_idx]
        ctx_emb = extractor.predict(np.array([ids]), verbose=0)[0]
        
        sim_matrix = cosine_similarity(ctx_emb)
        self.fig_sim.clear()
        ax = self.fig_sim.add_subplot(111)
        cax = ax.imshow(sim_matrix, cmap='magma')
        ax.set_xticks(np.arange(len(words))); ax.set_yticks(np.arange(len(words)))
        ax.set_xticklabels(words, rotation=90); ax.set_yticklabels(words)
        ax.set_title(f"Contextual Embedding Cosine Similarity (Layer {layer_idx})")
        self.fig_sim.colorbar(cax, ax=ax)
        self.canvas_sim.draw()

    def compute_analogy(self):
        if not self.llm.model: return
        a, b, c = self.input_a.text().strip(), self.input_b.text().strip(), self.input_c.text().strip()
        if not a or not b or not c: return
        weights = self.llm.embedding_layer.token_emb.get_weights()[0]
        id_a = self.llm.tokenizer.encode(a).ids[0]; id_b = self.llm.tokenizer.encode(b).ids[0]; id_c = self.llm.tokenizer.encode(c).ids[0]
        target_vec = (weights[id_a] - weights[id_b] + weights[id_c]).reshape(1, -1)
        sims = cosine_similarity(target_vec, weights)[0]
        ignore_ids = [id_a, id_b, id_c, 0, 1]
        top_idx = [i for i in np.argsort(sims)[::-1] if i not in ignore_ids][:5]
        self.table_math.setRowCount(0)
        for idx in top_idx:
            w = self.llm.tokenizer.decode([idx]); row = self.table_math.rowCount()
            self.table_math.insertRow(row)
            self.table_math.setItem(row, 0, QTableWidgetItem(w)); self.table_math.setItem(row, 1, QTableWidgetItem(f"{sims[idx]:.4f}"))

    def compute_contextual(self):
        if not self.llm.model: return
        word = self.input_word.text().strip(); s1 = self.input_s1.text().strip(); s2 = self.input_s2.text().strip()
        weights = self.llm.embedding_layer.token_emb.get_weights()[0]
        word_id = self.llm.tokenizer.encode(word).ids[0]
        static_vec = weights[word_id].reshape(1, -1)
        static_sim = cosine_similarity(static_vec, static_vec)[0][0]
        
        layer_idx = int(self.combo_ctx_layer.currentText().split(" ")[1])
        extractor = self.llm.context_extractors[layer_idx]
        
        def get_ctx_vec(sentence):
            enc = self.llm.tokenizer.encode(sentence); ids = enc.ids
            try: target_idx = ids.index(word_id)
            except ValueError: return None
            if len(ids) > self.llm.seq_length:
                start = max(0, target_idx - self.llm.seq_length // 2); ids = ids[start : start + self.llm.seq_length]; target_idx = target_idx - start
            else:
                padding = [0] * (self.llm.seq_length - len(ids)); ids = padding + ids; target_idx = target_idx + len(padding)
            ctx_emb = extractor.predict(np.array([ids]), verbose=0)
            return ctx_emb[0, target_idx, :].reshape(1, -1)
        vec1 = get_ctx_vec(s1); vec2 = get_ctx_vec(s2)
        if vec1 is None or vec2 is None:
            self.label_ctx.setText("Error: Word not found in sentence."); return
        ctx_sim = cosine_similarity(vec1, vec2)[0][0]
        self.label_ctx.setText(f"<b>Static Sim (Initial Emb):</b> {static_sim:.4f} (always 1.0)\n<b>Contextual Sim (Layer {layer_idx}):</b> {ctx_sim:.4f}")

    def plot_embeddings_2d(self):
        if not self.llm.model: return
        weights = self.llm.embedding_layer.token_emb.get_weights()[0]; valid_tokens = weights[2:] 
        vocab = self.llm.tokenizer.get_vocab(); id_to_word = {v: k for k, v in vocab.items()}
        labels = [id_to_word[i+2] for i in range(len(valid_tokens))]
        if self.radio_tsne.isChecked():
            self.log("Computing t-SNE..."); QApplication.processEvents()
            reducer = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto'); emb_2d = reducer.fit_transform(valid_tokens); title = "Token Embeddings (t-SNE 2D Projection)"
        else:
            reducer = PCA(n_components=2); emb_2d = reducer.fit_transform(valid_tokens); title = "Token Embeddings (PCA 2D Projection)"
        self.fig_emb.clear(); ax = self.fig_emb.add_subplot(111)
        ax.scatter(emb_2d[:, 0], emb_2d[:, 1], color='#03a9f4', s=10)
        for i in np.random.choice(len(labels), 50, replace=False): ax.annotate(labels[i], (emb_2d[i, 0], emb_2d[i, 1]), fontsize=8, color='white')
        ax.set_title(title); self.canvas_emb.draw()

    def plot_norms(self):
        if not self.llm.model: return
        weights = self.llm.embedding_layer.token_emb.get_weights()[0]; norms = np.linalg.norm(weights[2:], axis=1)
        self.fig_emb.clear(); ax = self.fig_emb.add_subplot(111)
        ax.hist(norms, bins=50, color='#e91e63', edgecolor='black')
        ax.set_title("Distribution of Vector L2 Norms"); ax.set_xlabel("L2 Norm"); ax.set_ylabel("Number of Tokens")
        self.canvas_emb.draw()

    def plot_model_weights(self):
        if not self.llm.model: return
        all_weights = []
        for w in self.llm.model.weights:
            if len(w.shape) == 2: all_weights.extend(w.numpy().flatten())
        self.fig_emb.clear(); ax = self.fig_emb.add_subplot(111)
        ax.hist(all_weights, bins=100, color='#4caf50', edgecolor='black')
        ax.set_title("Histogram of All Model Weights (2D Matrices)")
        ax.set_xlabel("Weight Value"); ax.set_ylabel("Frequency")
        self.canvas_emb.draw()

    def plot_freq_vs_norm(self):
        if not self.llm.model: return
        weights = self.llm.embedding_layer.token_emb.get_weights()[0]
        norms = np.linalg.norm(weights, axis=1); freqs = self.llm.id_to_freq
        self.fig_emb.clear(); ax = self.fig_emb.add_subplot(111)
        valid_idx = [i for i in range(2, len(norms)) if freqs[i] > 0]
        valid_norms = norms[valid_idx]; valid_freqs = freqs[valid_idx]
        ax.scatter(np.log(valid_freqs), valid_norms, color='#03a9f4', s=10, alpha=0.6)
        ax.set_title("Token Frequency (log) vs. Vector L2 Norm")
        ax.set_xlabel("Token Frequency (log scale)"); ax.set_ylabel("L2 Norm (Vector Length)")
        self.canvas_emb.draw()

    # --- TAB 8: MATH VISUALIZERS ---
    def setup_tab_math(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("<h2>Step 7: Math Visualizers</h2>"))
        grp_pe = QGroupBox("Sinusoidal Positional Encodings")
        pe_layout = QVBoxLayout()
        pe_form = QFormLayout()
        self.spin_pe_dim = QSpinBox(); self.spin_pe_dim.setRange(16, 128); self.spin_pe_dim.setSingleStep(16); self.spin_pe_dim.setValue(64)
        pe_form.addRow("Embedding Dimension:", self.spin_pe_dim)
        self.spin_pe_len = QSpinBox(); self.spin_pe_len.setRange(10, 100); self.spin_pe_len.setValue(24)
        pe_form.addRow("Sequence Length:", self.spin_pe_len)
        pe_layout.addLayout(pe_form)
        btn_plot_pe = QPushButton("Visualize Positional Encoding Matrix"); btn_plot_pe.clicked.connect(self.plot_positional)
        pe_layout.addWidget(btn_plot_pe)
        self.fig_pe = Figure(); self.canvas_pe = FigureCanvas(self.fig_pe)
        pe_layout.addWidget(self.canvas_pe)
        grp_pe.setLayout(pe_layout); layout.addWidget(grp_pe)
        
        grp_act = QGroupBox("Activation Functions & Derivatives")
        act_layout = QVBoxLayout()
        act_form = QHBoxLayout()
        act_form.addWidget(QLabel("Activation:"))
        self.combo_act_math = QComboBox(); self.combo_act_math.addItems(["gelu", "relu", "elu", "tanh", "sigmoid"])
        act_form.addWidget(self.combo_act_math); act_form.addStretch()
        btn_plot_act = QPushButton("Plot Activation"); btn_plot_act.clicked.connect(self.plot_activation)
        act_form.addWidget(btn_plot_act); act_layout.addLayout(act_form)
        self.fig_act = Figure(); self.canvas_act = FigureCanvas(self.fig_act)
        act_layout.addWidget(self.canvas_act); grp_act.setLayout(act_layout); layout.addWidget(grp_act)
        
        grp_rope = QGroupBox("RoPE (Rotary Positional Embedding) Concept")
        rope_layout = QVBoxLayout()
        rope_info = QTextBrowser()
        rope_info.setHtml("""
            <h3 style="color:#03a9f4;">Rotary Positional Embeddings (RoPE)</h3>
            <p>Used in modern LLMs (LLaMA, Mistral), RoPE rotates the Query and Key vectors based on their position.</p>
            <p>For a 2D embedding $(x_1, x_2)$ at position $m$, RoPE applies a rotation matrix:</p>
            <p style="font-family:monospace; background:#333; padding:5px;">$x' = x_1 \\cos(m\\theta) - x_2 \\sin(m\\theta)$</p>
            <p style="font-family:monospace; background:#333; padding:5px;">$y' = x_1 \\sin(m\\theta) + x_2 \\cos(m\\theta)$</p>
            <p>This ensures that the dot product between Query and Key naturally decays with relative distance, without needing to add explicit positional vectors.</p>
        """)
        rope_layout.addWidget(rope_info)
        rope_form = QFormLayout()
        self.spin_rope_m = QSpinBox(); self.spin_rope_m.setRange(1, 50); self.spin_rope_m.setValue(10)
        rope_form.addRow("Max Position (m):", self.spin_rope_m)
        self.spin_rope_theta = QDoubleSpinBox(); self.spin_rope_theta.setRange(0.1, 10.0); self.spin_rope_theta.setValue(1.0)
        rope_form.addRow("Base Theta:", self.spin_rope_theta)
        rope_layout.addLayout(rope_form)
        btn_plot_rope = QPushButton("Visualize RoPE Rotation"); btn_plot_rope.clicked.connect(self.plot_rope)
        rope_layout.addWidget(btn_plot_rope)
        self.fig_rope = Figure(); self.canvas_rope = FigureCanvas(self.fig_rope)
        rope_layout.addWidget(self.canvas_rope)
        grp_rope.setLayout(rope_layout); layout.addWidget(grp_rope)
        
        layout.addLayout(self.create_nav_layout(8))
        self.tabs.addTab(tab, "8. Math Visualizers")

    def plot_positional(self):
        d_model = self.spin_pe_dim.value(); length = self.spin_pe_len.value()
        pe = np.zeros((length, d_model)); position = np.arange(0, length, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2).astype(np.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term); pe[:, 1::2] = np.cos(position * div_term)
        self.fig_pe.clear(); ax = self.fig_pe.add_subplot(111); cax = ax.imshow(pe, cmap='coolwarm', aspect='auto')
        ax.set_xlabel("Embedding Dimension"); ax.set_ylabel("Sequence Position"); ax.set_title("Sinusoidal Positional Encoding")
        self.fig_pe.colorbar(cax, ax=ax); self.canvas_pe.draw()

    def plot_activation(self):
        act_name = self.combo_act_math.currentText()
        x = np.linspace(-5, 5, 100)
        if act_name == "gelu": y = 0.5 * x * (1 + tf.math.erf(x / np.sqrt(2.0)).numpy())
        elif act_name == "relu": y = np.maximum(0, x)
        elif act_name == "elu": y = np.where(x > 0, x, np.exp(x) - 1)
        elif act_name == "tanh": y = np.tanh(x)
        elif act_name == "sigmoid": y = 1 / (1 + np.exp(-x))
        dy = np.gradient(y, x)
        self.fig_act.clear(); ax = self.fig_act.add_subplot(111)
        ax.plot(x, y, label=f"{act_name}(x)", color='#03a9f4', linewidth=2)
        ax.plot(x, dy, label="Derivative (Gradient)", color='#e91e63', linestyle='--')
        ax.axhline(0, color='white', linewidth=0.5); ax.axvline(0, color='white', linewidth=0.5)
        ax.legend(); ax.set_title(f"{act_name.upper()} Activation Function")
        ax.set_xlabel("Input (x)"); ax.set_ylabel("Output (y)"); ax.grid(True, alpha=0.3)
        self.canvas_act.draw()

    def plot_rope(self):
        m = self.spin_rope_m.value()
        theta = self.spin_rope_theta.value()
        
        self.fig_rope.clear()
        ax = self.fig_rope.add_subplot(111, projection='polar')
        
        for pos in range(1, m+1, max(1, m//5)):
            angle = pos * theta
            ax.plot([angle, angle], [0, 1], marker='o', label=f"Pos {pos}")
            
        ax.set_title("RoPE Rotation of Vectors (Polar Plot)")
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        self.canvas_rope.draw()

    # --- TAB 9: PERSISTENCE ---
    def setup_tab_persistence(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("<h2>Step 8: Save / Load Model</h2>"))
        h_layout = QHBoxLayout()
        btn_save = QPushButton("Save Model & Tokenizer"); btn_save.clicked.connect(self.save_model); h_layout.addWidget(btn_save)
        btn_load = QPushButton("Load Model & Tokenizer"); btn_load.clicked.connect(self.load_model); h_layout.addWidget(btn_load)
        layout.addLayout(h_layout); layout.addStretch()
        self.tabs.addTab(tab, "9. Save/Load")

    def save_model(self):
        if not self.llm.model: return
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory to Save Model")
        if not dir_path: return
        try:
            self.llm.model.save_weights(os.path.join(dir_path, "model_weights.weights.h5"))
            self.llm.tokenizer.save(os.path.join(dir_path, "tokenizer.json"))
            with open(os.path.join(dir_path, "config.json"), 'w') as f: json.dump(self.llm.config, f)
            QMessageBox.information(self, "Success", "Model saved!")
        except Exception as e: QMessageBox.critical(self, "Error", f"Failed: {e}")

    def load_model(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory containing Model")
        if not dir_path: return
        try:
            with open(os.path.join(dir_path, "config.json"), 'r') as f: config = json.load(f)
            self.llm.tokenizer = Tokenizer.from_file(os.path.join(dir_path, "tokenizer.json")); self.llm.vocab_size = self.llm.tokenizer.get_vocab_size()
            self.llm.build_model(config['embed_dim'], config['num_heads'], config['ff_dim'], config['num_layers'], config['activation'], config['optimizer'], config['clip_norm'], config.get('use_pre_ln', False), config.get('pos_type', 'learned'), config.get('dropout_rate', 0.1), config.get('weight_decay', 0.01), config.get('epochs', 15), config.get('warmup_steps', 20), config.get('batch_size', 32), config.get('freeze_emb', False))
            self.llm.model.load_weights(os.path.join(dir_path, "model_weights.weights.h5"))
            
            self.combo_att_layer.clear()
            self.combo_ctx_layer.clear()
            self.combo_sim_layer.clear()
            self.combo_ctx_layer.addItem("Static (0)")
            self.combo_sim_layer.addItem("Static (0)")
            for i in range(len(self.llm.transformer_blocks)):
                self.combo_att_layer.addItem(f"Layer {i+1}")
                self.combo_ctx_layer.addItem(f"Layer {i+1}")
                self.combo_sim_layer.addItem(f"Layer {i+1}")
            self.combo_ctx_layer.addItem(f"Final ({len(self.llm.transformer_blocks)})")
            self.combo_sim_layer.addItem(f"Final ({len(self.llm.transformer_blocks)})")
            
            QMessageBox.information(self, "Success", "Model loaded!")
        except Exception as e: QMessageBox.critical(self, "Error", f"Failed: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())