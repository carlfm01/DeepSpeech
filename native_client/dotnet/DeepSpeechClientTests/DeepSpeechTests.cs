using DeepSpeechClient.Interfaces;
using DeepSpeechClient.Models;
using NAudio.Wave;
using System;
using System.IO;
using Xunit;

namespace DeepSpeechClient.Tests
{
    public class DeepSpeechTests
    {
        private const float LM_ALPHA = 0.75f;
        private const float LM_BETA = 1.85f;
        private const int BEAM_WIDTH = 500;

        [Theory]
        [InlineData("output_graph.pbmm")]
        [InlineData("output_graph.pb")]
        public void CreateModel_LoadValidModel_CreatesTheModel(string modelPath)
        {
            Action createModel = () =>
            {
                using (IDeepSpeech sttClient = new DeepSpeech(modelPath, BEAM_WIDTH))
                {
                }
            };

            var exception = Record.Exception(createModel);

            Assert.Null(exception);
        }

        [Theory]
        [InlineData("output_graph.pbmm", "")]
        [InlineData("output_graph.pb", "non-existent.binary")]
        [InlineData("output_graph.pb", "")]
        [InlineData("output_graph.pbmm", "non-existent.binary")]
        public void EnableDecoderWithLM_InvalidLanguageModelPath_ThrowsFileNotFoundException(string modelPath,
            string languageModelPath)
        {
            Action enableDecoderWithLM = () =>
            {
                using (IDeepSpeech sttClient = new DeepSpeech(modelPath, BEAM_WIDTH))
                {

                    sttClient.EnableDecoderWithLM(languageModelPath,
                        TestDefaults.TriePath, LM_ALPHA, LM_BETA);
                }
            };

            Assert.Throws<FileNotFoundException>(enableDecoderWithLM);
        }

        [Theory]
        [InlineData("output_graph.pbmm", "")]
        [InlineData("output_graph.pb", "triee")]
        [InlineData("output_graph.pb", "")]
        [InlineData("output_graph.pbmm", "triee")]
        public void EnableDecoderWithLM_InvalidTriePath_ThrowsFileNotFoundException(string modelPath,
            string triePath)
        {
            Action enableDecoderWithLM = () =>
            {
                using (IDeepSpeech sttClient = new DeepSpeech(modelPath, BEAM_WIDTH))
                {

                    sttClient.EnableDecoderWithLM(TestDefaults.LanguageModelPath,
                        triePath, LM_ALPHA, LM_BETA);
                }
            };

            Assert.Throws<FileNotFoundException>(enableDecoderWithLM);
        }

        [Fact]
        public void
            EnableDecoderWithLM_EnableDecoderWithLMWithoutCreatingAcousticModel_ThrowsInvalidOperationException()
        {
            Action enableDecoderWithLM = () =>
            {
                using (IDeepSpeech sttClient = new DeepSpeech(modelPath, BEAM_WIDTH))
                {
                    sttClient.EnableDecoderWithLM(TestDefaults.LanguageModelPath,
                        TestDefaults.TriePath, LM_ALPHA, LM_BETA);
                }
            };

            Assert.Throws<InvalidOperationException>(enableDecoderWithLM);
        }

        [Fact]
        public void FeedAudioContent_FeedAudioWithoutCreatingAcousticModel_ThrowsInvalidOperationException()
        {
            Action feedAudioContent = () =>
            {
                using (IDeepSpeech sttClient = new DeepSpeech(modelPath, BEAM_WIDTH))
                {
                    sttClient.FeedAudioContent(new DeepSpeechStream(),
                        new short[] {0, 0, 0, 0, 0, 0}, 0);
                }
            };

            Assert.Throws<InvalidOperationException>(feedAudioContent);
        }

        [Fact]
        public void FinishStream_FinishStreamWithoutCreatingAcousticModel_ThrowsInvalidOperationException()
        {
            Action finishStream = () =>
            {
                using (IDeepSpeech sttClient = new DeepSpeech(modelPath, BEAM_WIDTH))
                {
                    sttClient.FinishStream(new DeepSpeechStream());
                }
            };

            Assert.Throws<InvalidOperationException>(finishStream);
        }

        [Fact]
        public void GetModelSampleRate_GetModelSampleRateWithoutCreatingAcousticModel_ThrowsInvalidOperationException()
        {
            Action getModelSampleRate = () =>
            {
                using (IDeepSpeech sttClient = new DeepSpeech(modelPath, BEAM_WIDTH))
                {
                    sttClient.GetModelSampleRate();
                }
            };

            Assert.Throws<InvalidOperationException>(getModelSampleRate);
        }

        [Theory]
        [InlineData("output_graph.pbmm")]
        [InlineData("output_graph.pb")]
        public void GetModelSampleRate_AcousticModelCreatedProperly_CorrectSampleRate(string modelPath)
        {
            int sampleRate = -1;

            using (IDeepSpeech sttClient = new DeepSpeech(modelPath, BEAM_WIDTH))
            {

                sampleRate = sttClient.GetModelSampleRate();
            }

            Assert.Equal(16000, sampleRate);
        }

        [Theory]
        [InlineData("output_graph.pbmm", false)]
        [InlineData("output_graph.pb", false)]
        [InlineData("output_graph.pbmm", true)]
        [InlineData("output_graph.pb", true)]
        public void SpeechToTextWithMetadata_FeedLDCFile_CorrectTranscription(string modelPath,
            bool loadLanguageModel)
        {
            Metadata sttResult;

            using (IDeepSpeech sttClient = new DeepSpeech(modelPath, BEAM_WIDTH))
            {

                if (loadLanguageModel)
                {
                    sttClient.EnableDecoderWithLM(TestDefaults.LanguageModelPath, TestDefaults.TriePath, LM_ALPHA,
                        LM_BETA);
                }

                var waveBuffer = new WaveBuffer(File.ReadAllBytes(TestDefaults.LDCWavFilePath));
                using (var waveInfo = new WaveFileReader(TestDefaults.LDCWavFilePath))
                {
                    sttResult = sttClient.SpeechToTextWithMetadata(waveBuffer.ShortBuffer,
                        Convert.ToUInt32(waveBuffer.MaxSize / 2));
                }
            }

            Assert.NotNull(sttResult);
            Assert.True(sttResult.Confidence > 0);
            Assert.NotEmpty(sttResult.Items);
            Assert.Equal(52, sttResult.Items.Length);
        }

        [Theory]
        [InlineData("output_graph.pbmm", false)]
        [InlineData("output_graph.pb", false)]
        [InlineData("output_graph.pbmm", true)]
        [InlineData("output_graph.pb", true)]
        public void FinishStream_CreateAndFinishStreamWithoutFeedingAnyData_EmptyTranscription(string modelPath,
            bool loadLanguageModel)
        {
            string sttResult = "not empty";

            using (IDeepSpeech sttClient = new DeepSpeech(modelPath, BEAM_WIDTH))
            {

                if (loadLanguageModel)
                {
                    sttClient.EnableDecoderWithLM(TestDefaults.LanguageModelPath, TestDefaults.TriePath, LM_ALPHA,
                        LM_BETA);
                }

                var stream = sttClient.CreateStream();
                sttResult = sttClient.FinishStream(stream);
            }

            Assert.Equal(string.Empty, sttResult);
        }

        [Theory]
        [InlineData("output_graph.pbmm", false)]
        [InlineData("output_graph.pb", false)]
        [InlineData("output_graph.pbmm", true)]
        [InlineData("output_graph.pb", true)]
        public void FinishStream_FeedSilenceAudio_ReturnsEmptyTranscription(string modelPath,
            bool loadLanguageModel)
        {
            string sttResult = "not empty";

            using (IDeepSpeech sttClient = new DeepSpeech(modelPath, BEAM_WIDTH))
            {

                if (loadLanguageModel)
                {
                    sttClient.EnableDecoderWithLM(TestDefaults.LanguageModelPath, TestDefaults.TriePath, LM_ALPHA,
                        LM_BETA);
                }

                var waveBuffer = new WaveBuffer(File.ReadAllBytes("empty.wav"));
                var stream = sttClient.CreateStream();
                sttClient.FeedAudioContent(stream, waveBuffer.ShortBuffer, Convert.ToUInt32(waveBuffer.MaxSize / 2));

                sttResult = sttClient.FinishStream(stream);
            }

            Assert.Equal(string.Empty, sttResult);
        }

        [Theory]
        [InlineData("output_graph.pbmm", false)]
        [InlineData("output_graph.pb", false)]
        [InlineData("output_graph.pbmm", true)]
        [InlineData("output_graph.pb", true)]
        public void FinishStream_FeedLDCAudioFileSingleStream_ReturnsCorrectTranscription(string modelPath,
            bool loadLanguageModel)
        {
            string sttResult = "";

            using (IDeepSpeech sttClient = new DeepSpeech(modelPath, BEAM_WIDTH))
            {

                if (loadLanguageModel)
                {
                    sttClient.EnableDecoderWithLM(TestDefaults.LanguageModelPath, TestDefaults.TriePath, LM_ALPHA,
                        LM_BETA);
                }

                var waveBuffer = new WaveBuffer(File.ReadAllBytes(TestDefaults.LDCWavFilePath));
                var stream = sttClient.CreateStream();
                sttClient.FeedAudioContent(stream, waveBuffer.ShortBuffer, Convert.ToUInt32(waveBuffer.MaxSize / 2));

                sttResult = sttClient.FinishStream(stream);
            }

            Assert.Equal(TestDefaults.LDCTranscription, sttResult);
        }

        [Theory]
        [InlineData("output_graph.pbmm", false)]
        [InlineData("output_graph.pb", false)]
        [InlineData("output_graph.pbmm", true)]
        [InlineData("output_graph.pb", true)]
        public void FreeStream_FreeStreamWithoutFeedingData_FinishesTheStream(string modelPath,
            bool loadLanguageModel)
        {
            DeepSpeechStream stream;
            using (IDeepSpeech sttClient = new DeepSpeech(modelPath, BEAM_WIDTH))
            {

                if (loadLanguageModel)
                {
                    sttClient.EnableDecoderWithLM(TestDefaults.LanguageModelPath, TestDefaults.TriePath, LM_ALPHA,
                        LM_BETA);
                }

                stream = sttClient.CreateStream();

                sttClient.FreeStream(stream);
            }

            Assert.False(stream.Active);
        }

        [Theory]
        [InlineData("output_graph.pbmm", false)]
        [InlineData("output_graph.pb", false)]
        [InlineData("output_graph.pbmm", true)]
        [InlineData("output_graph.pb", true)]
        public void FreeStream_FreeStreamFeedingLDCFile_FinishesTheStream(string modelPath,
            bool loadLanguageModel)
        {
            DeepSpeechStream stream;
            using (IDeepSpeech sttClient = new DeepSpeech(modelPath, BEAM_WIDTH))
            {

                if (loadLanguageModel)
                {
                    sttClient.EnableDecoderWithLM(TestDefaults.LanguageModelPath, TestDefaults.TriePath, LM_ALPHA,
                        LM_BETA);
                }

                var waveBuffer = new WaveBuffer(File.ReadAllBytes(TestDefaults.LDCWavFilePath));
                stream = sttClient.CreateStream();
                sttClient.FeedAudioContent(stream, waveBuffer.ShortBuffer,
                    Convert.ToUInt32(waveBuffer.MaxSize / 2));

                sttClient.FreeStream(stream);
            }

            Assert.False(stream.Active);
        }

        [Theory]
        [InlineData("output_graph.pbmm", false)]
        [InlineData("output_graph.pb", false)]
        [InlineData("output_graph.pbmm", true)]
        [InlineData("output_graph.pb", true)]
        public void FinishStream_FeedLDCAudioFileMultiStream_ReturnsCorrectTranscription(string modelPath,
            bool loadLanguageModel)
        {
            string sttResultOne = "";
            string sttResultTwo = "";

            using (IDeepSpeech sttClient = new DeepSpeech(modelPath, BEAM_WIDTH))
            {

                if (loadLanguageModel)
                {
                    sttClient.EnableDecoderWithLM(TestDefaults.LanguageModelPath, TestDefaults.TriePath, LM_ALPHA,
                        LM_BETA);
                }

                var waveBuffer = new WaveBuffer(File.ReadAllBytes(TestDefaults.LDCWavFilePath));
                var streamOne = sttClient.CreateStream();
                var streamTwo = sttClient.CreateStream();

                sttClient.FeedAudioContent(streamOne, waveBuffer.ShortBuffer, Convert.ToUInt32(waveBuffer.MaxSize / 2));
                sttClient.FeedAudioContent(streamTwo, waveBuffer.ShortBuffer, Convert.ToUInt32(waveBuffer.MaxSize / 2));

                sttResultOne = sttClient.FinishStream(streamOne);
                sttResultTwo = sttClient.FinishStream(streamTwo);
            }

            Assert.Equal(TestDefaults.LDCTranscription, sttResultOne);
            Assert.Equal(TestDefaults.LDCTranscription, sttResultTwo);
        }

        [Theory]
        [InlineData("output_graph.pbmm", false)]
        [InlineData("output_graph.pb", false)]
        [InlineData("output_graph.pbmm", true)]
        [InlineData("output_graph.pb", true)]
        public void FinishStreamWithMetadata_FeedSilenceAudio_ReturnsEmptyMetadata(string modelPath,
            bool loadLanguageModel)
        {
            Metadata sttResult = null;

            using (IDeepSpeech sttClient = new DeepSpeech(modelPath, BEAM_WIDTH))
            {

                if (loadLanguageModel)
                {
                    sttClient.EnableDecoderWithLM(TestDefaults.LanguageModelPath, TestDefaults.TriePath, LM_ALPHA,
                        LM_BETA);
                }

                var waveBuffer = new WaveBuffer(File.ReadAllBytes("empty.wav"));
                var stream = sttClient.CreateStream();
                sttClient.FeedAudioContent(stream, waveBuffer.ShortBuffer, Convert.ToUInt32(waveBuffer.MaxSize / 2));
                sttResult = sttClient.FinishStreamWithMetadata(stream);
            }

            Assert.NotNull(sttResult);
            Assert.Empty(sttResult.Items);
        }

        [Fact]
        public void
            SpeechToTextWithMetadata_SpeechToTextWithMetadataWithoutCreatingAcousticModel_ThrowsInvalidOperationException()
        {
            Action speechToTextWithMetadata = () =>
            {
                using (IDeepSpeech sttClient = new DeepSpeech(modelPath, BEAM_WIDTH))
                {
                    sttClient.SpeechToTextWithMetadata(new short[] {0, 0, 0, 0, 0, 0}, 0);
                }
            };

            Assert.Throws<InvalidOperationException>(speechToTextWithMetadata);
        }

        [Fact]
        public void IntermediateDecode_IntermediateDecodeWithoutCreatingStream_ThrowsInvalidOperationException()
        {
            Action intermediateDecode = () =>
            {
                using (IDeepSpeech sttClient = new DeepSpeech(modelPath, BEAM_WIDTH))
                {
                    sttClient.IntermediateDecode(new DeepSpeechStream());
                }
            };

            Assert.Throws<InvalidOperationException>(intermediateDecode);
        }

        [Fact]
        public void FinishStream_FinishStreamWithoutCreatingStream_ThrowsInvalidOperationException()
        {
            Action finishStream = () =>
            {
                using (IDeepSpeech sttClient = new DeepSpeech(modelPath, BEAM_WIDTH))
                {
                    sttClient.FinishStream(new DeepSpeechStream());
                }
            };

            Assert.Throws<InvalidOperationException>(finishStream);
        }

        [Fact]
        public void
            FinishStreamWithMetadata_FinishStreamWithMetadataWithoutCreatingStream_ThrowsInvalidOperationException()
        {
            Action finishStreamWithMetadata = () =>
            {
                using (IDeepSpeech sttClient = new DeepSpeech(modelPath, BEAM_WIDTH))
                {
                    sttClient.FinishStreamWithMetadata(new DeepSpeechStream());
                }
            };

            Assert.Throws<InvalidOperationException>(finishStreamWithMetadata);
        }

        [Fact]
        public void FeedAudioContent_FeedAudioContentWithoutCreatingStream_ThrowsInvalidOperationException()
        {
            Action finishStreamWithMetadata = () =>
            {
                using (IDeepSpeech sttClient = new DeepSpeech(modelPath, BEAM_WIDTH))
                {
                    sttClient.FeedAudioContent(new DeepSpeechStream(),
                        new short[] {0, 0, 0, 0, 0, 0}, 0);
                }
            };

            Assert.Throws<InvalidOperationException>(finishStreamWithMetadata);
        }

        [Fact]
        public void CreateStream_CreateStreamWithoutCreatingAcousticModel_ThrowsInvalidOperationException()
        {
            Action createStream = () =>
            {
                using (IDeepSpeech sttClient = new DeepSpeech(modelPath, BEAM_WIDTH))
                {
                    sttClient.CreateStream();
                }
            };

            Assert.Throws<InvalidOperationException>(createStream);
        }

        [Fact]
        public void SpeechToText_SpeechToTextWithoutCreatingAcousticModel_ThrowsInvalidOperationException()
        {
            Action speechToText = () =>
            {
                using (IDeepSpeech sttClient = new DeepSpeech(modelPath, BEAM_WIDTH))
                {
                    sttClient.SpeechToText(new short[] {0, 0, 0, 0, 0, 0}, 0);
                }
            };

            Assert.Throws<InvalidOperationException>(speechToText);
        }

        [Theory]
        [InlineData("output_graph.pbmm", false)]
        [InlineData("output_graph.pb", false)]
        [InlineData("output_graph.pbmm", true)]
        [InlineData("output_graph.pb", true)]
        public void SpeechToText_FeedSilenceAudio_ReturnsEmptyTranscription(string modelPath, bool loadLanguageModel)
        {
            string sttResult = "not empty";

            using (IDeepSpeech sttClient = new DeepSpeech(modelPath, BEAM_WIDTH))
            {

                if (loadLanguageModel)
                {
                    sttClient.EnableDecoderWithLM(TestDefaults.LanguageModelPath, TestDefaults.TriePath, LM_ALPHA,
                        LM_BETA);
                }

                var waveBuffer = new WaveBuffer(File.ReadAllBytes("empty.wav"));
                sttResult = sttClient.SpeechToText(waveBuffer.ShortBuffer, Convert.ToUInt32(waveBuffer.MaxSize / 2));
            }

            Assert.Equal(string.Empty, sttResult);
        }

        [Theory]
        [InlineData("output_graph.pbmm", false)]
        [InlineData("output_graph.pb", false)]
        [InlineData("output_graph.pbmm", true)]
        [InlineData("output_graph.pb", true)]
        public void SpeechToText_FeedLDCAudio_ReturnsCorrectTranscription(string modelPath, bool loadLanguageModel)
        {
            string sttResult;

            using (IDeepSpeech sttClient = new DeepSpeech(modelPath, BEAM_WIDTH))
            {

                if (loadLanguageModel)
                {
                    sttClient.EnableDecoderWithLM(TestDefaults.LanguageModelPath, TestDefaults.TriePath, LM_ALPHA,
                        LM_BETA);
                }

                var waveBuffer = new WaveBuffer(File.ReadAllBytes(TestDefaults.LDCWavFilePath));

                sttResult = sttClient.SpeechToText(waveBuffer.ShortBuffer, Convert.ToUInt32(waveBuffer.MaxSize / 2));
            }

            Assert.Equal(TestDefaults.LDCTranscription, sttResult);
        }

        [Fact]
        public void FinishStreamWithMetadata_FinishStreamWithMetadataWithoutCreatingAcousticModel_ThrowsInvalidOperationException()
        {
            Action finishStream = () =>
            {
                using (IDeepSpeech sttClient = new DeepSpeech(modelPath, BEAM_WIDTH))
                {
                    sttClient.FinishStreamWithMetadata(new DeepSpeechStream());
                }
            };

            Assert.Throws<InvalidOperationException>(finishStream);
        }

        [Theory]
        [InlineData("a.pbmm")]
        [InlineData("")]
        public void CreateModel_InvalidModelPath_ThrowsFileNotFoundException(string modelPath)
        {
            Action createModel = () =>
            {
                using (IDeepSpeech sttClient = new DeepSpeech(modelPath, BEAM_WIDTH))
                {

                }
            };

            Assert.Throws<FileNotFoundException>(createModel);
        }

        [Fact]
        public void PrintVersions_WithoutLoadingAcousticModel_DoesNotThrowsException()
        {
            Action printVersions = () =>
            {
                using (IDeepSpeech sttClient = new DeepSpeech(modelPath, BEAM_WIDTH))
                {
                    sttClient.PrintVersions();
                }
            };
            var exception = Record.Exception(printVersions);

            Assert.Null(exception);
        }

        [Fact]
        public void IntermediateDecode_IntermediateDecodeWithoutCreatingAcousticModel_ThrowsInvalidOperationException()
        {
            Action intermediateDecode = () =>
            {
                using (IDeepSpeech sttClient = new DeepSpeech(modelPath, BEAM_WIDTH))
                {
                    sttClient.IntermediateDecode(new DeepSpeechStream());
                }
            };

            Assert.Throws<InvalidOperationException>(intermediateDecode);
        }

        [Theory]
        [InlineData("output_graph.pbmm", false)]
        [InlineData("output_graph.pb", false)]
        [InlineData("output_graph.pbmm", true)]
        [InlineData("output_graph.pb", true)]
        public void IntermediateDecode_FeedEmptyAudio_ReturnsEmptyTranscription(string modelPath,
            bool loadLanguageModel)
        {
            string intermediateDecode = "not empty";
            using (IDeepSpeech sttClient = new DeepSpeech(modelPath, BEAM_WIDTH))
            {

                if (loadLanguageModel)
                {
                    sttClient.EnableDecoderWithLM(TestDefaults.LanguageModelPath, TestDefaults.TriePath, LM_ALPHA,
                        LM_BETA);
                }

                var stream = sttClient.CreateStream();
                var waveBuffer = new WaveBuffer(File.ReadAllBytes("empty.wav"));
                sttClient.FeedAudioContent(stream, waveBuffer.ShortBuffer, Convert.ToUInt32(waveBuffer.MaxSize / 2));
                intermediateDecode = sttClient.IntermediateDecode(stream);
            }

            Assert.Equal(string.Empty, intermediateDecode);
        }

        [Theory]
        [InlineData("output_graph.pbmm", false)]
        [InlineData("output_graph.pb", false)]
        [InlineData("output_graph.pbmm", true)]
        [InlineData("output_graph.pb", true)]
        public void IntermediateDecodeAndFinishStream_FeedLDCAudio_ReturnsCorrectTranscriptionForIntermediateDecodeAndFinishStream(
                string modelPath, bool loadLanguageModel)
        {
            string intermediateDecode = null;
            string finishResult = null;
            using (IDeepSpeech sttClient = new DeepSpeech(modelPath, BEAM_WIDTH))
            {

                if (loadLanguageModel)
                {
                    sttClient.EnableDecoderWithLM(TestDefaults.LanguageModelPath, TestDefaults.TriePath, LM_ALPHA,
                        LM_BETA);
                }

                var stream = sttClient.CreateStream();
                var waveBuffer = new WaveBuffer(File.ReadAllBytes(TestDefaults.LDCWavFilePath));
                sttClient.FeedAudioContent(stream, waveBuffer.ShortBuffer,
                    Convert.ToUInt32(waveBuffer.MaxSize / 2));

                intermediateDecode = sttClient.IntermediateDecode(stream);
                finishResult = sttClient.FinishStream(stream);
            }

            Assert.NotNull(intermediateDecode);
            Assert.NotNull(finishResult);
            Assert.Equal(TestDefaults.LDCTranscription, finishResult);
            Assert.Equal(TestDefaults.LDCTranscription, intermediateDecode);
        }
    }
}