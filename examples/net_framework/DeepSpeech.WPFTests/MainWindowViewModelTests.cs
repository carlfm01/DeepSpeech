using CSCore;
using CSCore.Codecs;
using CSCore.CoreAudioAPI;
using CSCore.SoundOut;
using DeepSpeech.WPF.ViewModels;
using DeepSpeechClient.Interfaces;
using System;
using System.Linq;
using System.Threading;
using Xunit;
using Xunit.Abstractions;

namespace DeepSpeech.WPF.Tests
{
    [Collection("Models collection")]
    public class MainWindowViewModelTests
    {

        public MainWindowViewModelTests()
        {
        }

        private const float LM_ALPHA = 0.75f;
        private const float LM_BETA = 1.85f;
        private const int BEAM_WIDTH = 500;
        private static DeepSpeechClient.DeepSpeech CreateSttModel(bool loadLanguageModel, string modelPath)
        {
            var sttClient = new DeepSpeechClient.DeepSpeech(modelPath, BEAM_WIDTH);
             
            if (loadLanguageModel)
            {
                sttClient.EnableDecoderWithLM(TestDefaults.LanguageModelPath,
                      TestDefaults.TriePath, LM_ALPHA, LM_BETA);
            }
            return sttClient;
        }

        [Theory()]
        [InlineData(true, "output_graph.pbmm")]
        [InlineData(false, "output_graph.pbmm")]
        [InlineData(true, "output_graph.pb")]
        [InlineData(false, "output_graph.pb")]
        public void InferenceFromFileCommand_UseLDCAudioFile_GeneratesCorrectTranscription(bool loadLanguageModel, string modelPath)
        {
            IDeepSpeech sttClient = CreateSttModel(loadLanguageModel, modelPath);

            string transcriptionResult = string.Empty;
            var viewModel = new MainWindowViewModel(sttClient)
            {
                AudioFilePath = TestDefaults.LDCWavFilePath
            };

            viewModel.InferenceFromFileCommand.ExecuteAsync().GetAwaiter().GetResult();
            transcriptionResult = viewModel.Transcription;
            sttClient.Dispose();

            Assert.EndsWith(TestDefaults.LDCTranscription, transcriptionResult);
        }

        [Theory()]
        [InlineData(true, "output_graph.pbmm")]
        [InlineData(false, "output_graph.pbmm")]
        [InlineData(true, "output_graph.pb")]
        [InlineData(false, "output_graph.pb")]
        public void InferenceFromFileCommand_NonExistentAudioFile_GeneratesEmptyTranscription(bool loadLanguageModel, string modelPath)
        {
            IDeepSpeech sttClient = CreateSttModel(loadLanguageModel, modelPath);
            var viewModel = new MainWindowViewModel(sttClient)
            {
                AudioFilePath = "non-existent.wav"
            };

            viewModel.InferenceFromFileCommand.ExecuteAsync().GetAwaiter().GetResult();

            Assert.Equal(string.Empty, viewModel.Transcription);
        }

        [Theory()]
        [InlineData(true, "output_graph.pbmm")]
        [InlineData(false, "output_graph.pbmm")]
        [InlineData(true, "output_graph.pb")]
        [InlineData(false, "output_graph.pb")]
        public void StartAndStopStreamCommands_NoPlaybackToRecord_GeneratesEmptyTranscription(bool loadLanguageModel, string modelPath)
        {
            IDeepSpeech sttClient = CreateSttModel(loadLanguageModel, modelPath);
            var viewModel = new MainWindowViewModel(sttClient);

            viewModel.StartRecordingCommand.Execute(null);
            viewModel.StopRecordingCommand.ExecuteAsync().GetAwaiter().GetResult();
            sttClient.Dispose();

            Assert.Equal(string.Empty, viewModel.Transcription);
        }

        [Theory()]
        [InlineData(true, "output_graph.pbmm")]
        [InlineData(false, "output_graph.pbmm")]
        [InlineData(true, "output_graph.pb")]
        [InlineData(false, "output_graph.pb")]
        public void StartAndStopStreamCommands_RecordLDCAudioFile_GeneratesCorrectTranscription(bool loadLanguageModel, string modelPath)
        {
            IDeepSpeech sttClient = CreateSttModel(loadLanguageModel, modelPath);
            var viewModel = new MainWindowViewModel(sttClient);
            viewModel.SelectedDevice = viewModel.AvailableRecordDevices.First(
                x => x.DataFlow == DataFlow.Render);

            using (IWaveSource soundSource = CodecFactory.Instance.GetCodec(TestDefaults.LDCWavFilePath))
            {
                using (ISoundOut soundOut = WasapiOut.IsSupportedOnCurrentPlatform ?
                    (ISoundOut)new WasapiOut { Device = viewModel.SelectedDevice }
                    : new DirectSoundOut { Device = new Guid(viewModel.SelectedDevice.DeviceID) })
                {
                    soundOut.Initialize(soundSource);
                    viewModel.StartRecordingCommand.Execute(null);
                    soundOut.Play();

                    Thread.Sleep(2955);

                    soundOut.Stop();
                }
            }

            viewModel.StopRecordingCommand.ExecuteAsync().GetAwaiter().GetResult();
            sttClient.Dispose();

            Assert.Equal(TestDefaults.LDCTranscription, viewModel.Transcription);
        }
    }
}