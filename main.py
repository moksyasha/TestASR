import nemo.collections.asr as nemo_asr
import torch, torchaudio
import os
from pyannote.audio import Pipeline
from pydub import AudioSegment


def respond_to_text(model, audio_path):
    transcription = model.transcribe([audio_path])[0][0]
    if "привет я разработчик" == transcription:
        return "сегодня выходной"
    elif "я сегодня не приду домой" == transcription:
        return "Ну и катись отсюда"
    else:
        return "Я не понял вас"


def first(asr_model, tts_model):
    # stt
    output_phrase1 = respond_to_text(asr_model, "./data/hi.wav")
    output_phrase2 = respond_to_text(asr_model, "./data/today.wav")
    output_phrase3 = respond_to_text(asr_model, "./data/trash.wav")

    #tts
    audio = tts_model.apply_tts(text=output_phrase1,
                        speaker="kseniya",
                        sample_rate=24000)
    torchaudio.save('./result/test_1.wav',
                  audio.unsqueeze(0),
                  sample_rate=24000)
    
    audio = tts_model.apply_tts(text=output_phrase2,
                    speaker="kseniya",
                    sample_rate=24000)
    torchaudio.save('./result/test_2.wav',
                  audio.unsqueeze(0),
                  sample_rate=24000)
    
    audio = tts_model.apply_tts(text=output_phrase3,
                    speaker="kseniya",
                    sample_rate=24000)
    torchaudio.save('./result/test_3.wav',
                  audio.unsqueeze(0),
                  sample_rate=24000)


def second(asr_model):
    # diarization
    # https://huggingface.co/pyannote/speaker-diarization
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token="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")

    mix_audio = AudioSegment.from_wav("./data/mix.wav")
    diarization = pipeline("./data/mix.wav", num_speakers=2)
    segments = []
    start = 0

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        mix_audio_clip = mix_audio[start*1000:turn.end*1000]
        mix_audio_clip.export('./temp.wav', format="wav")
        transcription = asr_model.transcribe(['./temp.wav'])[0][0]
        segments.append((speaker, turn.start, turn.end, transcription))
        start = turn.end
    
    print(segments)
    # [('SPEAKER_01', 0.03096875, 2.37659375, 'привет как жизнь чем занимаешься'), 
    #  ('SPEAKER_00', 2.37659375, 5.785343750000001, 'привет у меня все хорошо работаю сегодня'), 
    #  ('SPEAKER_01', 5.90346875, 7.709093750000001, 'может сходим куда нибудь вечером'), 
    #  ('SPEAKER_00', 7.709093750000001, 8.974718750000001, 'да конечно можно')]

def main():
    # # stt
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/stt_ru_conformer_transducer_large")
    
    # # tts
    torch.set_num_threads(4)
    local_file = 'model_tts.pt'

    if not os.path.isfile(local_file):
        torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v4_ru.pt',
                                    local_file)  

    tts_model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tts_model.to(device)

    first(asr_model, tts_model)
    second(asr_model)


if __name__ == "__main__":
    main()