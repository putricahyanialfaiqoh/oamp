import os
import time
import random
from pathlib import Path

import pygame
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

class MotivationAudioPlayer:
    def __init__(self):
        pygame.mixer.init()
        self.last_played_time = 0
        self.cooldown_seconds = 5  # Cooldown 5 detik agar audio tidak overlap
        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID", "XB0fDUnXU5powFXDhCwa")
        self.voice_cache_tag = os.getenv("ELEVENLABS_VOICE_CACHE_TAG", "charlotte")
        self.offline_only = os.getenv("ELEVENLABS_OFFLINE_ONLY", "false").lower() == "true"
        api_key = os.getenv("ELEVENLABS_API_KEY")
        self.elevenlabs = ElevenLabs(api_key=api_key) if api_key else None
        
        if self.elevenlabs is None:
            print("[AUDIO ROBOT]: ELEVENLABS_API_KEY belum diatur; audio tidak dapat dibuat")

        self.speech_map = {
            "sad": [
                "Gapapa, yuk coba lagi pelan-pelan.",
                "Kamu sudah hebat mau mencoba. Istirahat sebentar, ya.",
                "Jangan sedih, kita selesaikan sedikit demi sedikit.",
                "Belum berhasil bukan berarti tidak bisa. Yuk coba lagi!",
                "Ayo semangat! Satu langkah kecil lagi."
            ],
            "fear": [
                "Tenang aja, kamu pasti bisa.",
                "Yuk tarik napas dulu. Pelan-pelan aja.",
                "Tidak perlu takut, kita coba bersama-sama yuk.",
                "Kamu lebih berani dari yang kamu kira.",
                "Lihat polanya baik-baik. Kamu bisa melakukannya!"
            ],
            "angry": [
                "Yuk tarik napas dulu, pelan pelan aja nyelesainya.",
                "Tidak apa-apa kalau kesal. Istirahat sebentar, ya.",
                "Tenang, kamu masih punya kesempatan.",
                "Yuk fokus lagi. Kamu pasti bisa menyelesaikannya.",
                "Sudah lebih tenang? Ayo lanjut bermain!"
            ],
            "disgust": [
                "Yuk lanjut, tinggal sedikit lagi.",
                "Ayo, kamu pasti bisa menemukan jawabannya.",
                "Semangat, kita coba lagi yaa.",
                "Jangan nyerah, kamu sudah hampir berhasil.",
                "Lihat baik-baik, polanya."
            ],
            "happy": [
                "Wah, hebat! Kamu mengerjakannya dengan baik.",
                "Keren banget! Pertahankan semangatmu, ya.",
                "Mantap! Kamu semakin jago.",
                "Senyumnya bagus! Yuk lanjutkan permainan.",
                "Wah...kamu luar biasa."
            ],
            "surprise": [
                "Wah, ada kejutan! Tenang, kamu bisa.",
                "Kaget sedikit tidak apa-apa. Yuk lanjut lagi.",
                "Wah, keren! Sekarang kita lihat bagian berikutnya.",
                "Tenang saja, kamu siap untuk tantangan berikutnya.",
                "Yuk tarik napas, lalu kita lanjut bermain."
            ],
            "neutral": [
                "Ayo mulai, kamu pasti bisa!",
                "Yuk cari pola yang cocok.",
                "Bagus, teruskan seperti itu.",
                "Lihat polanya pelan-pelan, ya.",
                "Langkahmu sudah bagus. Yuk lanjut!"
            ]
        }

        self.invitation_text = "Yuk mulai bermain! Kamu pasti bisa!"

    def _play_text(self, text, filename):
        current_time = time.time()

        if current_time - self.last_played_time < self.cooldown_seconds:
            print("[AUDIO ROBOT]: Menunggu cooldown audio")
            return False

        print(f"[AUDIO ROBOT]: Memutar suara: {text}")
        cached_filename = filename
        if not os.path.exists(cached_filename):
            legacy_filename = filename.replace(
                f"{self.voice_cache_tag}_", "", 1
            )
            if os.path.exists(legacy_filename):
                cached_filename = legacy_filename
            elif self.offline_only or not self._generate_audio(text, filename):
                print(f"[AUDIO ROBOT]: Cache audio tidak ditemukan: {filename}")
                return False

        try:
            pygame.mixer.music.load(cached_filename)
        except pygame.error as error:
            print(f"[AUDIO ROBOT]: Cache audio rusak, membuat ulang: {error}")
            try:
                os.remove(filename)
            except OSError:
                pass

            if not self._generate_audio(text, filename):
                return False
            pygame.mixer.music.load(filename)

        pygame.mixer.music.play()
        self.last_played_time = current_time
        return True

    def _generate_audio(self, text, filename):
        if self.elevenlabs is None:
            print("[AUDIO ROBOT]: ELEVENLABS_API_KEY belum diatur")
            return False

        temporary_filename = f"{filename}.tmp"
        try:
            audio = self.elevenlabs.text_to_speech.convert(
                voice_id=self.voice_id,
                text=text,
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
                voice_settings={
                    "stability": 0.35,
                    "similarity_boost": 0.75,
                    "style": 0.40,
                    "use_speaker_boost": True,
                },
            )
            with open(temporary_filename, "wb") as audio_file:
                for chunk in audio:
                    audio_file.write(chunk)
            os.replace(temporary_filename, filename)
            return True
        except Exception as error:
            print(f"[AUDIO ROBOT]: Gagal membuat audio ElevenLabs: {error}")
            try:
                os.remove(temporary_filename)
            except OSError:
                pass
            return False

    def play_motivation_for_emotion(self, emotion):
        options = self.speech_map.get(emotion, self.speech_map["neutral"])
        
        # Pilih index dan teks secara acak
        index = random.randint(0, len(options) - 1)
        selected_text = options[index]
        
        # Cache per index agar tidak generate API berulang untuk opsi kalimat yang sama
        filename = f"elevenlabs_{self.voice_cache_tag}_{emotion}_{index}.mp3"
        return self._play_text(selected_text, filename)

    def play_invitation(self):
        return self._play_text(
            self.invitation_text,
            f"elevenlabs_{self.voice_cache_tag}_invitation.mp3"
        )