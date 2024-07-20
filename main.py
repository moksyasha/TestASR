import nemo.collections.asr as nemo_asr
import torch
import os


def respond_to_text(model, audio_path):
    transcription = model.transcribe([audio_path])[0][0]
    if "привет я разработчик" == transcription:
        return "сегодня выходной"
    elif "я сегодня не приду домой" == transcription:
        return "Ну и катись отсюда"
    else:
        return "Я не понял вас"


def main():
    # stt
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/stt_ru_conformer_transducer_large")

    output_phrase1 = respond_to_text(asr_model, "./hi.wav")
    output_phrase2 = respond_to_text(asr_model, "./today.wav")
    output_phrase3 = respond_to_text(asr_model, "./trash.wav")

    # tts
    torch.set_num_threads(4)
    local_file = 'model_tts.pt'

    if not os.path.isfile(local_file):
        torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v4_ru.pt',
                                    local_file)  

    tts_model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tts_model.to(device)

    audio = tts_model.save_wav(text=output_phrase3,
                        speaker="kseniya",
                        sample_rate=24000, )



if __name__ == "__main__":
    main()