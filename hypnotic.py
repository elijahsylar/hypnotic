"""
HYPNOTIC - Ultra-Responsive Fullscreen Audio Visualizer
One mesmerizing effect at a time. Instant audio response.

Controls:
    ← →     Switch effect
    T       Toggle study timer (45 min study / 15 min break)
    W       Toggle word display (speech-to-text)
    F       Toggle fullscreen
    ESC     Exit

Usage:
    python3 hypnotic.py                 # default input device
    python3 hypnotic.py <device_number> # a specific device (see --list)
    python3 hypnotic.py --list          # list input devices
    python3 hypnotic.py --demo          # synthetic signal, no audio device needed

First-time setup for speech recognition:
    pip install vosk spacy
    python -m spacy download en_core_web_sm
    
    # Download Vosk model (RECOMMENDED: use the larger model for accuracy)
    # Small (40MB, faster, less accurate):
    #   vosk-model-small-en-us-0.15
    # Large (1.8GB, slower, much more accurate):
    #   vosk-model-en-us-0.22
    # Extract to: ~/.vosk/model
"""

import numpy as np
import sounddevice as sd
import pygame
import sys
import math
import os
import json
from collections import deque

# Optional: Speech recognition
try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

# Optional: Smart keyword extraction
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

FPS = 60
FULLSCREEN = True

# Audio - small buffer = low latency = instant response
SAMPLE_RATE = 48000
BLOCK_SIZE = 512  # Small for responsiveness
FFT_SIZE = 512

# ============================================================================
# AUDIO ENGINE - ULTRA RESPONSIVE
# ============================================================================

class AudioEngine:
    """Low-latency audio capture with FFT."""
    
    def __init__(self):
        self.stream = None
        self.device_index = None
        
        # Current audio frame
        self.waveform = np.zeros(BLOCK_SIZE)
        self.fft = np.zeros(FFT_SIZE // 2)
        
        # Frequency bands - RAW, no smoothing for instant response
        self.bass = 0.0
        self.mids = 0.0  
        self.highs = 0.0
        self.energy = 0.0
        
        # Peak detection for punch
        self.peak = 0.0
        self.peak_decay = 0.0
        
        # Speech recognizer reference (set externally)
        self.speech_recognizer = None
        
    def audio_callback(self, indata, frames, time, status):
        """Audio thread callback: hand the first channel to process_block."""
        self.process_block(indata[:, 0].copy())

    def process_block(self, samples):
        """Analyze one block of mono samples - keep it fast."""
        # Raw waveform
        self.waveform = samples
        
        # Send to speech recognizer if available
        if self.speech_recognizer and self.speech_recognizer.enabled:
            self.speech_recognizer.process_audio(self.waveform)
        
        # FFT
        windowed = self.waveform * np.hanning(len(self.waveform))
        fft_raw = np.abs(np.fft.rfft(windowed))
        
        # Normalize
        max_val = np.max(fft_raw) + 0.001
        self.fft = fft_raw[:FFT_SIZE // 2] / max_val
        
        # Frequency bands - instant, no smoothing
        n = len(self.fft)
        self.bass = np.mean(self.fft[:int(n * 0.1)]) * 4
        self.mids = np.mean(self.fft[int(n * 0.1):int(n * 0.4)]) * 3
        self.highs = np.mean(self.fft[int(n * 0.4):]) * 2.5
        
        # Clamp
        self.bass = min(self.bass, 1.0)
        self.mids = min(self.mids, 1.0)
        self.highs = min(self.highs, 1.0)
        
        # Overall energy
        self.energy = (self.bass * 0.5 + self.mids * 0.3 + self.highs * 0.2)
        
        # Peak detection - for punchy response
        current_peak = max(self.bass, self.mids * 0.8)
        if current_peak > self.peak:
            self.peak = current_peak
        else:
            self.peak *= 0.85  # Fast decay
            
        self.peak_decay = self.peak
    
    def start(self, device_index):
        """Start audio stream."""
        try:
            self.stream = sd.InputStream(
                device=device_index,
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                callback=self.audio_callback,
                latency='low'
            )
            self.stream.start()
            return True
        except Exception as e:
            print(f"Audio error: {e}")
            return False
    
    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()

# ============================================================================
# SPEECH RECOGNITION
# ============================================================================

class SpeechRecognizer:
    """Real-time speech-to-text with intelligent keyword extraction using spaCy."""
    
    # Extensive list of common/filler words to skip - we only want TECHNICAL terms
    SKIP_WORDS = {
        # Common nouns that aren't really keywords
        'thing', 'things', 'stuff', 'way', 'ways', 'time', 'times', 'day', 'days',
        'year', 'years', 'people', 'person', 'man', 'men', 'woman', 'women',
        'child', 'children', 'place', 'places', 'part', 'parts', 'case', 'cases',
        'point', 'points', 'fact', 'facts', 'lot', 'lots', 'kind', 'kinds',
        'sort', 'sorts', 'type', 'types', 'bit', 'bits', 'end', 'ends',
        'hand', 'hands', 'head', 'eye', 'eyes', 'side', 'sides', 'home',
        'world', 'life', 'word', 'words', 'work', 'number', 'example',
        'order', 'course', 'something', 'nothing', 'anything', 'everything',
        'someone', 'anyone', 'everyone', 'nobody', 'none', 'ones',
        'today', 'tomorrow', 'yesterday', 'morning', 'evening', 'night',
        'week', 'month', 'hour', 'minute', 'second', 'moment', 'while',
        'beginning', 'middle', 'rest', 'top', 'bottom', 'level', 'line',
        'area', 'room', 'group', 'member', 'members', 'result', 'results',
        'idea', 'ideas', 'question', 'questions', 'answer', 'answers',
        'problem', 'problems', 'issue', 'issues', 'reason', 'reasons',
        'name', 'names', 'change', 'changes', 'difference', 'differences',
        'information', 'data', 'story', 'stories', 'history', 'news',
        'money', 'price', 'cost', 'value', 'amount', 'size', 'age',
        'water', 'food', 'air', 'fire', 'earth', 'nature', 'weather',
        'city', 'country', 'state', 'town', 'house', 'building', 'street',
        'car', 'door', 'window', 'table', 'chair', 'book', 'paper', 'page',
        'body', 'face', 'heart', 'mind', 'brain', 'blood', 'skin', 'bone',
        'family', 'friend', 'friends', 'parent', 'parents', 'mother', 'father',
        'brother', 'sister', 'son', 'daughter', 'husband', 'wife', 'baby',
        'teacher', 'student', 'students', 'doctor', 'president', 'leader',
        'company', 'business', 'market', 'industry', 'service', 'product',
        'system', 'process', 'method', 'approach', 'practice', 'action',
        'power', 'control', 'force', 'effect', 'effects', 'cause', 'causes',
        'sense', 'feeling', 'thought', 'thoughts', 'experience', 'knowledge',
        'truth', 'reality', 'form', 'forms', 'matter', 'material', 'source',
        'center', 'edge', 'corner', 'surface', 'ground', 'base', 'view',
        'interest', 'attention', 'focus', 'goal', 'goals', 'purpose', 'plan',
        'step', 'steps', 'stage', 'stages', 'phase', 'period', 'term',
        'start', 'finish', 'success', 'failure', 'effort', 'attempt',
        'choice', 'decision', 'option', 'options', 'chance', 'opportunity',
        'risk', 'benefit', 'advantage', 'disadvantage', 'feature', 'aspect',
        'detail', 'details', 'factor', 'factors', 'element', 'elements',
        'role', 'position', 'situation', 'condition', 'conditions', 'status',
        'relationship', 'connection', 'link', 'pattern', 'structure', 'model',
        'image', 'picture', 'color', 'light', 'sound', 'voice', 'music',
        'game', 'play', 'show', 'movie', 'film', 'scene', 'character',
        'event', 'events', 'meeting', 'party', 'trip', 'visit', 'tour',
        'need', 'needs', 'want', 'wants', 'wish', 'hope', 'fear', 'love',
        'death', 'war', 'peace', 'freedom', 'justice', 'law', 'laws', 'rule',
        'rules', 'right', 'rights', 'wrong', 'truth', 'lie', 'lies',
        # Additional very common words
        'look', 'looks', 'way', 'thing', 'use', 'uses', 'set', 'sets',
        'call', 'calls', 'class', 'note', 'notes', 'list', 'video', 'audio',
    }
    
    # Technical/academic suffixes that indicate important terms
    TECHNICAL_SUFFIXES = (
        'tion', 'sion',  # distribution, regression
        'ment',          # measurement, experiment  
        'ity', 'ty',     # probability, complexity
        'ism',           # algorithm, mechanism
        'ology', 'ics',  # biology, statistics
        'ical', 'ive',   # theoretical, derivative
        'ous', 'eous',   # continuous, homogeneous
        'able', 'ible',  # variable, divisible
        'ence', 'ance',  # variance, significance
        'ant', 'ent',    # determinant, coefficient
        'ure',           # structure, procedure
        'ary', 'ory',    # binary, trajectory
    )
    
    def __init__(self):
        self.recognizer = None
        self.model = None
        self.nlp = None  # spaCy model
        self.enabled = False
        self.available = VOSK_AVAILABLE
        
        # Buffer for accumulating words before NLP processing
        self.word_buffer = []
        self.last_process_time = 0
        self.process_interval = 600  # Process every 600ms for more context
        
        # Track recently shown words to avoid repetition
        self.recent_shown = deque(maxlen=20)
        
        # Keywords to display
        self.keywords = deque(maxlen=5)
        self.current_partial = ""
        
    def is_technical_word(self, word):
        """Check if word looks like a technical/academic term."""
        word_lower = word.lower()
        
        # Must be 5+ characters for technical terms
        if len(word_lower) < 5:
            return False
        
        # Skip if in our extensive skip list
        if word_lower in self.SKIP_WORDS:
            return False
        
        # Skip if recently shown
        if word_lower in self.recent_shown:
            return False
        
        # Check for technical suffixes - strong indicator
        for suffix in self.TECHNICAL_SUFFIXES:
            if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
                return True
        
        # Longer words (7+) that aren't common are likely technical
        if len(word_lower) >= 7:
            return True
        
        return False
    
    def load_model(self):
        """Load Vosk and spaCy models."""
        if not VOSK_AVAILABLE:
            print("⚠️  Vosk not installed. Run: pip install vosk")
            return False
        
        # Load spaCy for keyword extraction
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("✅ spaCy NLP loaded (smart keyword extraction)")
            except OSError:
                print("⚠️  spaCy model not found. Run: python -m spacy download en_core_web_sm")
                print("   Falling back to basic keyword detection")
                self.nlp = None
        else:
            print("⚠️  spaCy not installed. Run: pip install spacy")
            print("   Falling back to basic keyword detection")
        
        # Check common Vosk model locations
        model_paths = [
            os.path.expanduser("~/.vosk/model"),
            os.path.expanduser("~/vosk-model"),
            "vosk-model",
            "model",
            os.environ.get("VOSK_MODEL_PATH", ""),
        ]
        
        # Also check for any vosk-model-* directory in current folder
        try:
            for item in os.listdir("."):
                if item.startswith("vosk-model"):
                    model_paths.insert(0, item)
        except:
            pass
        
        for path in model_paths:
            if path and os.path.exists(path):
                try:
                    print(f"📖 Loading speech model: {path}")
                    self.model = Model(path)
                    self.recognizer = KaldiRecognizer(self.model, SAMPLE_RATE)
                    self.recognizer.SetWords(True)
                    print("✅ Speech recognition ready!")
                    
                    # Check model size and warn if small
                    model_size = sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, _, filenames in os.walk(path)
                        for filename in filenames
                    )
                    if model_size < 100_000_000:  # Less than 100MB
                        print("💡 Tip: Using small model. For better accuracy, download:")
                        print("   vosk-model-en-us-0.22 (1.8GB)")
                    
                    return True
                except Exception as e:
                    print(f"❌ Failed to load model: {e}")
        
        print("⚠️  No Vosk model found.")
        print("   Download from: https://alphacephei.com/vosk/models")
        print("   For best accuracy: vosk-model-en-us-0.22 (1.8GB)")
        print("   For speed: vosk-model-small-en-us-0.15 (40MB)")
        print("   Extract to: ~/.vosk/model")
        return False
    
    def extract_keywords_spacy(self, text):
        """Use spaCy to extract meaningful technical keywords."""
        if not self.nlp or not text.strip():
            return []
        
        doc = self.nlp(text)
        keywords = []
        
        for token in doc:
            word = token.text
            word_lower = word.lower()
            
            # Skip short words, skip list, and recently shown
            if len(word_lower) < 5:
                continue
            if word_lower in self.SKIP_WORDS:
                continue
            if word_lower in self.recent_shown:
                continue
            
            score = 0
            
            # NOUNS and PROPER NOUNS are candidates
            if token.pos_ in ('NOUN', 'PROPN'):
                score = len(word_lower)  # Base score = length
                
                # Boost proper nouns significantly (technical terms, names)
                if token.pos_ == 'PROPN':
                    score += 5
                
                # Boost technical suffixes
                for suffix in self.TECHNICAL_SUFFIXES:
                    if word_lower.endswith(suffix):
                        score += 8
                        break
                
                # Boost longer words (more specific = more important)
                if len(word_lower) >= 8:
                    score += 4
                elif len(word_lower) >= 6:
                    score += 2
                
                # Only keep high-scoring words
                if score >= 8:
                    keywords.append((word_lower, score, word))
        
        # Sort by score, return only the BEST keyword
        keywords.sort(key=lambda x: x[1], reverse=True)
        
        # Return just 1 keyword - the most important one
        if keywords:
            return [keywords[0][2]]
        return []
    
    def extract_keywords_basic(self, text):
        """Basic keyword extraction without spaCy - focus on technical terms."""
        words = text.split()
        keywords = []
        
        for word in words:
            clean = word.lower().strip('.,!?;:')
            
            # Skip short, common, or recently shown
            if len(clean) < 5:
                continue
            if clean in self.SKIP_WORDS:
                continue
            if clean in self.recent_shown:
                continue
            
            score = len(clean)
            
            # Must have a technical suffix OR be quite long
            has_technical_suffix = False
            for suffix in self.TECHNICAL_SUFFIXES:
                if clean.endswith(suffix):
                    score += 8
                    has_technical_suffix = True
                    break
            
            # If no technical suffix, must be 7+ chars
            if not has_technical_suffix and len(clean) < 7:
                continue
            
            if len(clean) >= 8:
                score += 4
            
            if score >= 10:
                keywords.append((clean, score, word))
        
        keywords.sort(key=lambda x: x[1], reverse=True)
        
        if keywords:
            return [keywords[0][2]]
        return []
    
    def extract_keywords_fallback(self, text):
        """Fallback: grab the best noun available with looser criteria."""
        if self.nlp:
            doc = self.nlp(text)
            candidates = []
            
            for token in doc:
                word = token.text
                word_lower = word.lower()
                
                # Looser: 4+ chars, not in skip list, not recently shown
                if len(word_lower) < 4:
                    continue
                if word_lower in self.SKIP_WORDS:
                    continue
                if word_lower in self.recent_shown:
                    continue
                
                # Accept any noun
                if token.pos_ in ('NOUN', 'PROPN'):
                    score = len(word_lower)
                    if token.pos_ == 'PROPN':
                        score += 3
                    candidates.append((score, word))
            
            if candidates:
                candidates.sort(reverse=True)
                return [candidates[0][1]]
        else:
            # Basic fallback: longest word not in skip list
            words = text.split()
            candidates = []
            
            for word in words:
                clean = word.lower().strip('.,!?;:')
                if len(clean) >= 4 and clean not in self.SKIP_WORDS and clean not in self.recent_shown:
                    candidates.append((len(clean), word))
            
            if candidates:
                candidates.sort(reverse=True)
                return [candidates[0][1]]
        
        return []
    
    def process_audio(self, audio_data):
        """Process audio chunk for speech recognition."""
        if not self.recognizer or not self.enabled:
            return
        
        # Convert float32 to int16 for Vosk
        audio_int16 = (audio_data * 32767).astype(np.int16).tobytes()
        
        if self.recognizer.AcceptWaveform(audio_int16):
            # Final result - extract text
            result = json.loads(self.recognizer.Result())
            text = result.get("text", "")
            
            if text:
                self.word_buffer.append(text)
            
            self.current_partial = ""
        else:
            # Partial result - don't show partials, too distracting
            self.current_partial = ""
        
        # Process accumulated text periodically
        now = pygame.time.get_ticks()
        if self.word_buffer and (now - self.last_process_time) > self.process_interval:
            self.last_process_time = now
            
            # Combine buffer into text
            full_text = " ".join(self.word_buffer)
            self.word_buffer.clear()
            
            # Extract keywords
            if self.nlp:
                keywords = self.extract_keywords_spacy(full_text)
            else:
                keywords = self.extract_keywords_basic(full_text)
            
            # Fallback: if no keywords found and it's been a while, lower standards
            if not keywords:
                time_since_last = now - (self.keywords[-1]["time"] if self.keywords else 0)
                if time_since_last > 4000:  # 4 seconds without a word
                    keywords = self.extract_keywords_fallback(full_text)
            
            # Add to display queue and track as shown
            for kw in keywords:
                self.keywords.append({
                    "word": kw,
                    "time": now,
                })
                self.recent_shown.append(kw.lower())
    
    def get_display_words(self, max_age_ms=3000):
        """Get keywords to display."""
        now = pygame.time.get_ticks()
        display = []
        
        # list(): the speech thread may append while we iterate
        for w in list(self.keywords):
            age = now - w["time"]
            if age < max_age_ms:
                fade = 1.0 - (age / max_age_ms)
                display.append({
                    "word": w["word"],
                    "fade": fade,
                })
        
        return display
    
    def toggle(self):
        """Toggle speech recognition on/off."""
        if not self.available:
            print("⚠️  Vosk not available")
            return False
        
        if not self.model:
            if not self.load_model():
                return False
        
        self.enabled = not self.enabled
        status = "ON" if self.enabled else "OFF"
        print(f"🎤 Word display: {status}")
        return self.enabled


class WordRenderer:
    """Renders recognized keywords - clean and minimal."""
    
    def __init__(self):
        self.fonts = {}
    
    def get_font(self, size):
        """Cache fonts by size."""
        if size not in self.fonts:
            self.fonts[size] = pygame.font.Font(None, size)
        return self.fonts[size]
    
    def draw(self, screen, speech, audio, W, H, time_counter):
        """Draw recognized keywords on screen - one at a time, big and centered."""
        if not speech.enabled:
            return
        
        words = speech.get_display_words(max_age_ms=2500)
        
        if not words and not speech.current_partial:
            return
        
        # Show the most recent keyword BIG and centered
        if words:
            # Get the newest word
            newest = words[-1]
            word = newest["word"].upper()  # Uppercase for impact
            fade = newest["fade"]
            
            # Size based on fade - bigger when new
            base_size = int(min(W, H) * 0.15)
            size = int(base_size * (0.7 + fade * 0.3))
            
            font = self.get_font(size)
            
            # Color - synced with visualizer
            hue = (time_counter * 0.002) % 1.0
            brightness = 0.4 + fade * 0.6
            color = hsv_to_rgb(hue, 0.5, brightness)
            
            # Render dead center
            text_surf = font.render(word, True, color)
            text_rect = text_surf.get_rect(center=(W // 2, H // 2))
            
            # Glow effect for fresh words
            if fade > 0.6:
                glow_size = int(size * 1.05)
                glow_font = self.get_font(glow_size)
                glow_color = hsv_to_rgb(hue, 0.3, fade * 0.3)
                glow_surf = glow_font.render(word, True, glow_color)
                glow_rect = glow_surf.get_rect(center=(W // 2, H // 2))
                screen.blit(glow_surf, glow_rect)
            
            screen.blit(text_surf, text_rect)
            
            # Show previous word smaller above (if exists and recent enough)
            if len(words) >= 2:
                prev = words[-2]
                if prev["fade"] > 0.3:
                    prev_word = prev["word"].lower()
                    prev_size = int(base_size * 0.35)
                    prev_font = self.get_font(prev_size)
                    prev_color = hsv_to_rgb(hue, 0.3, prev["fade"] * 0.4)
                    prev_surf = prev_font.render(prev_word, True, prev_color)
                    prev_rect = prev_surf.get_rect(center=(W // 2, H // 2 - int(H * 0.12)))
                    screen.blit(prev_surf, prev_rect)
        
        # Show partial word (being spoken) even smaller
        elif speech.current_partial:
            partial_size = int(min(W, H) * 0.08)
            partial_font = self.get_font(partial_size)
            pulse = 0.5 + 0.3 * math.sin(time_counter * 0.2)
            partial_color = hsv_to_rgb(0.55, 0.4, pulse)
            partial_surf = partial_font.render(speech.current_partial, True, partial_color)
            partial_rect = partial_surf.get_rect(center=(W // 2, H // 2))
            screen.blit(partial_surf, partial_rect)

# ============================================================================
# EFFECTS
# ============================================================================

class RadialBurst:
    """
    Lines radiating from center - each line = frequency bin.
    Classic but hypnotic when done right.
    """
    name = "Radial Burst"
    
    def draw(self, screen, audio, W, H, time):
        cx, cy = W // 2, H // 2
        
        # Number of lines based on screen size
        num_lines = min(180, len(audio.fft))
        fft = audio.fft[:num_lines]
        
        # Base radius pulses with bass
        base_radius = 50 + audio.bass * 100
        max_length = min(W, H) * 0.45
        
        for i, val in enumerate(fft):
            angle = (i / num_lines) * 2 * math.pi - math.pi / 2
            
            # Length based on frequency magnitude
            length = base_radius + val * max_length * (1 + audio.peak_decay * 0.5)
            
            # End point
            x = cx + math.cos(angle) * length
            y = cy + math.sin(angle) * length
            
            # Inner point (slight gap in center)
            inner = base_radius * 0.3
            x0 = cx + math.cos(angle) * inner
            y0 = cy + math.sin(angle) * inner
            
            # Color - frequency based with time shift
            hue = (i / num_lines + time * 0.001) % 1.0
            color = hsv_to_rgb(hue, 0.8, 0.3 + val * 0.7)
            
            # Thickness based on energy
            thickness = max(1, int(2 + audio.energy * 4))
            
            pygame.draw.line(screen, color, (int(x0), int(y0)), (int(x), int(y)), thickness)
        
        # Center glow on bass hits
        if audio.bass > 0.3:
            glow_radius = int(40 + audio.bass * 60)
            glow_alpha = int(audio.bass * 150)
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (255, 255, 255, glow_alpha), 
                             (glow_radius, glow_radius), glow_radius)
            screen.blit(glow_surf, (cx - glow_radius, cy - glow_radius))


class WaveformRing:
    """
    Waveform wrapped in a circle - breathing, pulsing.
    """
    name = "Waveform Ring"
    
    def draw(self, screen, audio, W, H, time):
        cx, cy = W // 2, H // 2
        
        # Ring radius breathes with bass
        base_radius = min(W, H) * 0.25 + audio.bass * 80
        
        # Waveform displacement
        wave = audio.waveform
        num_points = len(wave)
        
        points_outer = []
        points_inner = []
        
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            
            # Waveform modulates the radius
            displacement = wave[i] * 150 * (1 + audio.energy)
            
            r_outer = base_radius + displacement
            r_inner = base_radius - displacement * 0.5
            
            points_outer.append((
                cx + math.cos(angle) * r_outer,
                cy + math.sin(angle) * r_outer
            ))
            points_inner.append((
                cx + math.cos(angle) * r_inner,
                cy + math.sin(angle) * r_inner
            ))
        
        # Draw outer ring
        if len(points_outer) > 2:
            # Color shifts with time and energy
            hue = (time * 0.002 + audio.mids * 0.3) % 1.0
            color = hsv_to_rgb(hue, 0.7, 0.8)
            pygame.draw.lines(screen, color, True, points_outer, max(2, int(3 + audio.bass * 4)))
        
        # Draw inner ring (complement color)
        if len(points_inner) > 2:
            hue2 = (hue + 0.5) % 1.0
            color2 = hsv_to_rgb(hue2, 0.6, 0.5)
            pygame.draw.lines(screen, color2, True, points_inner, 2)


class FrequencyTunnel:
    """
    Concentric rings moving toward you - 3D tunnel effect.
    Each ring = snapshot of FFT moving through z-space.
    """
    name = "Frequency Tunnel"
    
    def __init__(self):
        self.rings = []  # Store ring data: (fft_snapshot, z_depth, birth_time)
        self.spawn_timer = 0
    
    def draw(self, screen, audio, W, H, time):
        cx, cy = W // 2, H // 2
        
        # Spawn new rings based on beats
        self.spawn_timer += 1
        spawn_rate = max(3, int(10 - audio.energy * 8))  # Faster spawning with more energy
        
        if self.spawn_timer >= spawn_rate:
            self.spawn_timer = 0
            # Store FFT snapshot
            self.rings.append({
                'fft': audio.fft.copy(),
                'z': 0.0,
                'hue': (time * 0.002) % 1.0,
                'energy': audio.energy
            })
        
        # Update and draw rings
        new_rings = []
        for ring in self.rings:
            ring['z'] += 0.02 + audio.energy * 0.03  # Speed based on energy
            
            if ring['z'] < 1.0:  # Still visible
                new_rings.append(ring)
                
                # Perspective scale
                scale = 0.1 + ring['z'] * 0.9
                alpha = 1.0 - ring['z']
                
                # Draw the ring
                num_points = min(90, len(ring['fft']))
                fft = ring['fft'][:num_points]
                
                base_radius = min(W, H) * 0.4 * scale
                
                points = []
                for i in range(num_points):
                    angle = (i / num_points) * 2 * math.pi
                    # FFT modulates radius
                    r = base_radius * (0.8 + fft[i % len(fft)] * 0.4)
                    points.append((
                        cx + math.cos(angle) * r,
                        cy + math.sin(angle) * r
                    ))
                
                if len(points) > 2:
                    # Color fades as it approaches
                    color = hsv_to_rgb(ring['hue'], 0.8, alpha * 0.9)
                    thickness = max(1, int(3 * scale * (1 + ring['energy'])))
                    pygame.draw.lines(screen, color, True, 
                                    [(int(p[0]), int(p[1])) for p in points], thickness)
        
        self.rings = new_rings


class ParticleField:
    """
    Particles flowing outward from center, speed/color = frequency.
    Mesmerizing flow.
    """
    name = "Particle Flow"
    
    def __init__(self):
        self.particles = []
        self.max_particles = 800
    
    def draw(self, screen, audio, W, H, time):
        cx, cy = W // 2, H // 2
        max_dist = math.sqrt(cx**2 + cy**2)
        
        # Spawn particles from center
        spawn_count = int(5 + audio.energy * 20)
        for _ in range(spawn_count):
            if len(self.particles) < self.max_particles:
                angle = np.random.random() * 2 * math.pi
                speed = 2 + audio.bass * 8 + np.random.random() * 3
                
                # Frequency band determines color
                freq_band = np.random.random()
                if freq_band < 0.3:
                    hue = 0.0 + audio.bass * 0.1  # Red/orange for bass
                elif freq_band < 0.7:
                    hue = 0.45 + audio.mids * 0.15  # Cyan/green for mids
                else:
                    hue = 0.75 + audio.highs * 0.2  # Purple/pink for highs
                
                self.particles.append({
                    'x': cx,
                    'y': cy,
                    'vx': math.cos(angle) * speed,
                    'vy': math.sin(angle) * speed,
                    'hue': hue % 1.0,
                    'size': 2 + audio.peak_decay * 4,
                    'life': 1.0
                })
        
        # Update and draw
        new_particles = []
        for p in self.particles:
            # Move
            p['x'] += p['vx'] * (1 + audio.energy * 0.5)
            p['y'] += p['vy'] * (1 + audio.energy * 0.5)
            
            # Fade based on distance
            dist = math.sqrt((p['x'] - cx)**2 + (p['y'] - cy)**2)
            p['life'] = 1.0 - (dist / max_dist)
            
            if p['life'] > 0 and 0 < p['x'] < W and 0 < p['y'] < H:
                new_particles.append(p)
                
                # Draw
                color = hsv_to_rgb(p['hue'], 0.8, p['life'])
                size = max(1, int(p['size'] * p['life']))
                pygame.draw.circle(screen, color, (int(p['x']), int(p['y'])), size)
        
        self.particles = new_particles


class PulsingGeometry:
    """
    Sacred geometry that breathes with the music.
    """
    name = "Sacred Geometry"
    
    def draw(self, screen, audio, W, H, time):
        cx, cy = W // 2, H // 2
        
        # Base size pulses with bass
        base_size = min(W, H) * 0.3 * (0.8 + audio.bass * 0.4)
        
        # Draw multiple rotating polygons
        num_layers = 6
        
        for layer in range(num_layers):
            # Each layer: different size, rotation, sides
            progress = layer / num_layers
            
            size = base_size * (0.3 + progress * 0.7)
            rotation = time * 0.01 * (1 if layer % 2 == 0 else -1) + layer * 0.5
            rotation += audio.mids * 0.3  # Mids add rotation
            
            # Sides based on layer (triangle, square, pentagon, hexagon...)
            sides = 3 + layer
            
            # Draw polygon
            points = []
            for i in range(sides):
                angle = rotation + (i / sides) * 2 * math.pi
                # Vertices pulse with highs
                r = size * (1 + audio.highs * 0.2 * math.sin(i * 2 + time * 0.05))
                points.append((
                    cx + math.cos(angle) * r,
                    cy + math.sin(angle) * r
                ))
            
            # Color
            hue = (progress + time * 0.001) % 1.0
            brightness = 0.4 + (1 - progress) * 0.4 + audio.energy * 0.2
            color = hsv_to_rgb(hue, 0.7, brightness)
            
            if len(points) > 2:
                pygame.draw.lines(screen, color, True, 
                                [(int(p[0]), int(p[1])) for p in points],
                                max(1, int(2 + audio.bass * 3 * (1 - progress))))


class LiquidMorph:
    """
    Organic blob that morphs with audio using sine superposition.
    """
    name = "Liquid Morph"
    
    def draw(self, screen, audio, W, H, time):
        cx, cy = W // 2, H // 2
        
        # Base radius breathes
        base_r = min(W, H) * 0.25 * (0.8 + audio.bass * 0.4)
        
        num_points = 360
        points = []
        
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            
            # Superposition of sine waves for organic shape
            r = base_r
            
            # Bass = low frequency wobble
            r += math.sin(angle * 2 + time * 0.03) * 30 * audio.bass
            r += math.sin(angle * 3 - time * 0.02) * 25 * audio.bass
            
            # Mids = medium frequency detail
            r += math.sin(angle * 5 + time * 0.05) * 20 * audio.mids
            r += math.sin(angle * 7 - time * 0.04) * 15 * audio.mids
            
            # Highs = fine detail / shimmer
            r += math.sin(angle * 11 + time * 0.08) * 10 * audio.highs
            r += math.sin(angle * 13 + time * 0.1) * 8 * audio.highs
            
            # FFT direct modulation for extra responsiveness
            fft_idx = int((i / num_points) * len(audio.fft)) % len(audio.fft)
            r += audio.fft[fft_idx] * 50
            
            points.append((
                cx + math.cos(angle) * r,
                cy + math.sin(angle) * r
            ))
        
        # Fill with gradient-like effect using multiple layers
        for layer in range(3, 0, -1):
            scale = 0.7 + layer * 0.1
            layer_points = [(cx + (p[0] - cx) * scale, cy + (p[1] - cy) * scale) for p in points]
            
            hue = (time * 0.001 + layer * 0.1) % 1.0
            brightness = 0.2 + layer * 0.2 + audio.energy * 0.3
            color = hsv_to_rgb(hue, 0.8, brightness)
            
            pygame.draw.polygon(screen, color, [(int(p[0]), int(p[1])) for p in layer_points])
        
        # Outer edge
        hue_edge = (time * 0.001) % 1.0
        edge_color = hsv_to_rgb(hue_edge, 0.6, 0.9)
        pygame.draw.lines(screen, edge_color, True, 
                         [(int(p[0]), int(p[1])) for p in points], 
                         max(2, int(3 + audio.peak_decay * 4)))


class Oscilloscope:
    """
    Classic oscilloscope waveform - but stylized and reactive.
    """
    name = "Oscilloscope"
    
    def draw(self, screen, audio, W, H, time):
        cy = H // 2
        wave = audio.waveform
        num_points = len(wave)
        
        # Amplitude scales with energy
        amplitude = H * 0.35 * (0.5 + audio.energy * 0.5)
        
        points = []
        for i in range(num_points):
            x = (i / num_points) * W
            y = cy + wave[i] * amplitude
            points.append((x, y))
        
        if len(points) > 2:
            # Main wave
            hue = (time * 0.002) % 1.0
            color = hsv_to_rgb(hue, 0.8, 0.9)
            pygame.draw.lines(screen, color, False, 
                            [(int(p[0]), int(p[1])) for p in points],
                            max(2, int(3 + audio.bass * 4)))
            
            # Glow effect - draw slightly offset copies
            glow_color = hsv_to_rgb(hue, 0.5, 0.4)
            for offset in [-3, 3]:
                glow_points = [(p[0], p[1] + offset) for p in points]
                pygame.draw.lines(screen, glow_color, False,
                                [(int(p[0]), int(p[1])) for p in glow_points], 1)
        
        # Center line
        pygame.draw.line(screen, (30, 30, 40), (0, cy), (W, cy), 1)


# ============================================================================
# UTILITY
# ============================================================================

def hsv_to_rgb(h, s, v):
    """Fast HSV to RGB - always returns valid 0-255 values."""
    # Clamp inputs
    h = float(h) % 1.0
    s = max(0.0, min(1.0, float(s)))
    v = max(0.0, min(1.0, float(v)))
    
    if s == 0:
        r = g = b = int(v * 255)
        return (r, g, b)
    
    i = int(h * 6)
    f = (h * 6) - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    
    i = i % 6
    if i == 0: r, g, b = v, t, p
    elif i == 1: r, g, b = q, v, p
    elif i == 2: r, g, b = p, v, t
    elif i == 3: r, g, b = p, q, v
    elif i == 4: r, g, b = t, p, v
    else: r, g, b = v, p, q
    
    # Clamp output to valid range
    return (
        max(0, min(255, int(r * 255))),
        max(0, min(255, int(g * 255))),
        max(0, min(255, int(b * 255)))
    )


# ============================================================================
# MAIN
# ============================================================================

class DemoSignal:
    """Synthetic audio for --demo: a beat, a wandering tone and some noise,
    fed through the same analysis path as real input."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.t = 0.0

    def next_block(self):
        n = BLOCK_SIZE
        t = self.t + np.arange(n) / SAMPLE_RATE
        self.t = float(t[-1]) + 1.0 / SAMPLE_RATE
        beat = (np.sin(2 * np.pi * 2.0 * t) > 0.6) * np.sin(2 * np.pi * 60 * t)
        tone = 0.5 * np.sin(2 * np.pi * (440 + 220 * np.sin(0.3 * t)) * t)
        hiss = 0.1 * self.rng.standard_normal(n)
        return (beat + tone + hiss).astype(np.float32)


def parse_args(argv):
    import argparse
    p = argparse.ArgumentParser(description="HYPNOTIC - fullscreen audio visualizer")
    p.add_argument("device", nargs="?", type=int, default=None,
                   help="input device number (default: the system default input)")
    p.add_argument("--list", "-l", action="store_true", help="list input devices and exit")
    p.add_argument("--demo", action="store_true", help="use a synthetic signal instead of an audio device")
    p.add_argument("--windowed", action="store_true", help="start in a 1280x720 window")
    p.add_argument("--effect", type=int, default=0, help="starting effect index (0-6)")
    p.add_argument("--record", metavar="DIR", help="save each frame as a PNG in DIR (use with --frames)")
    p.add_argument("--frames", type=int, default=0, help="stop after this many frames (0 = run until ESC)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    print("=" * 50)
    print("  HYPNOTIC - Fullscreen Audio Visualizer")
    print("=" * 50)

    if args.list:
        devices = sd.query_devices()
        print("\nAudio Devices:")
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                print(f"  [{i}] {d['name']}")
        return
    device = args.device
    
    # Pygame init
    pygame.init()
    pygame.mouse.set_visible(False)
    
    # Get display info for fullscreen
    info = pygame.display.Info()
    W, H = info.current_w, info.current_h
    NATIVE_W, NATIVE_H = W, H   # F returns to this size, not the window size
    
    if FULLSCREEN and not args.windowed and not args.record:
        screen = pygame.display.set_mode((W, H), pygame.FULLSCREEN)
    else:
        W, H = 1280, 720
        screen = pygame.display.set_mode((W, H))
    
    pygame.display.set_caption("HYPNOTIC")
    clock = pygame.time.Clock()
    
    # Audio
    audio = AudioEngine()
    demo = DemoSignal() if args.demo else None
    if demo is None and not audio.start(device):
        print("Failed to start audio")
        return
    
    # Speech recognition
    speech = SpeechRecognizer()
    audio.speech_recognizer = speech
    word_renderer = WordRenderer()
    
    print(f"\n🎧 Audio: {'demo signal' if demo else (device if device is not None else 'default input')}")
    print(f"📺 Resolution: {W}x{H}")
    print("\nControls:")
    print("  ← →   Switch effect")
    print("  T     Toggle study timer (45 min on / 15 min break)")
    print("  W     Toggle word display (speech-to-text)")
    print("  F     Toggle fullscreen")
    print("  ESC   Exit\n")
    
    # Effects
    effects = [
        RadialBurst(),
        WaveformRing(),
        FrequencyTunnel(),
        ParticleField(),
        PulsingGeometry(),
        LiquidMorph(),
        Oscilloscope(),
    ]
    current_effect = args.effect % len(effects)
    if args.record:
        os.makedirs(args.record, exist_ok=True)
    
    print(f"🎨 Effect: {effects[current_effect].name}")
    
    time_counter = 0
    is_fullscreen = FULLSCREEN and not args.windowed and not args.record
    
    # Pomodoro timer
    timer_active = False
    timer_start = 0
    STUDY_TIME = 45 * 60  # 45 minutes in seconds
    BREAK_TIME = 15 * 60  # 15 minutes in seconds
    CYCLE_TIME = STUDY_TIME + BREAK_TIME
    
    # Font for timer display
    timer_font = pygame.font.Font(None, 48)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_RIGHT:
                    current_effect = (current_effect + 1) % len(effects)
                    print(f"🎨 Effect: {effects[current_effect].name}")
                elif event.key == pygame.K_LEFT:
                    current_effect = (current_effect - 1) % len(effects)
                    print(f"🎨 Effect: {effects[current_effect].name}")
                elif event.key == pygame.K_f:
                    is_fullscreen = not is_fullscreen
                    if is_fullscreen:
                        W, H = NATIVE_W, NATIVE_H
                        screen = pygame.display.set_mode((W, H), pygame.FULLSCREEN)
                    else:
                        screen = pygame.display.set_mode((1280, 720))
                        W, H = 1280, 720
                elif event.key == pygame.K_t:
                    timer_active = not timer_active
                    if timer_active:
                        timer_start = pygame.time.get_ticks()
                        print("⏱️  Timer started: 45 min study → 15 min break")
                    else:
                        print("⏱️  Timer stopped")
                elif event.key == pygame.K_w:
                    speech.toggle()
        
        time_counter += 1
        if demo is not None:
            audio.process_block(demo.next_block())
        
        # Calculate timer state
        is_break_time = False
        time_remaining = 0
        
        if timer_active:
            elapsed_ms = pygame.time.get_ticks() - timer_start
            elapsed_sec = elapsed_ms // 1000
            cycle_position = elapsed_sec % CYCLE_TIME
            
            if cycle_position < STUDY_TIME:
                # Study time
                is_break_time = False
                time_remaining = STUDY_TIME - cycle_position
            else:
                # Break time
                is_break_time = True
                time_remaining = CYCLE_TIME - cycle_position
        
        # Clear screen
        if is_break_time:
            # Break mode - black screen
            screen.fill((0, 0, 0))
            
            # Show break message
            break_text = timer_font.render("☕ BREAK TIME", True, (60, 60, 80))
            text_rect = break_text.get_rect(center=(W // 2, H // 2 - 30))
            screen.blit(break_text, text_rect)
            
            # Show time remaining
            mins = time_remaining // 60
            secs = time_remaining % 60
            time_str = f"{mins:02d}:{secs:02d}"
            time_text = timer_font.render(time_str, True, (80, 80, 100))
            time_rect = time_text.get_rect(center=(W // 2, H // 2 + 30))
            screen.blit(time_text, time_rect)
        else:
            # Study mode - visualizer
            fade = pygame.Surface((W, H))
            fade.fill((0, 0, 0))
            fade.set_alpha(30)
            screen.blit(fade, (0, 0))
            
            # Draw current effect
            effects[current_effect].draw(screen, audio, W, H, time_counter)
            
            # Draw recognized words
            word_renderer.draw(screen, speech, audio, W, H, time_counter)
            
            # Draw timer if active
            if timer_active:
                mins = time_remaining // 60
                secs = time_remaining % 60
                time_str = f"📖 {mins:02d}:{secs:02d}"
                time_text = timer_font.render(time_str, True, (50, 50, 70))
                screen.blit(time_text, (20, H - 50))
        
        pygame.display.flip()
        if args.record:
            pygame.image.save(screen, os.path.join(args.record, f"frame_{time_counter:05d}.png"))
        if args.frames and time_counter >= args.frames:
            running = False
        clock.tick(FPS)
    
    audio.stop()
    pygame.quit()
    print("\n✨ See you in the void.")


if __name__ == "__main__":
    main()
