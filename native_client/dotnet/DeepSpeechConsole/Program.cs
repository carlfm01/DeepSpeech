﻿using DeepSpeechClient;
using DeepSpeechClient.Interfaces;
using DeepSpeechClient.Models;
using NAudio.Wave;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace CSharpExamples
{
   
    public unsafe class Program
    {
        /// <summary>
        /// Get the value of an argurment.
        /// </summary>
        /// <param name="args">Argument list.</param>
        /// <param name="option">Key of the argument.</param>
        /// <returns>Value of the argument.</returns>
        static string GetArgument(IEnumerable<string> args, string option)
        => args.SkipWhile(i => i != option).Skip(1).Take(1).FirstOrDefault();

        static string MetadataToString(Metadata meta)
        {
            var nl = Environment.NewLine;
            string retval =
             Environment.NewLine + $"Recognized text: {string.Join("", meta?.Items?.Select(x => x.Character))} {nl}"
             + $"Prob: {meta?.Probability} {nl}"
             + $"Item count: {meta?.Items?.Length} {nl}"
             + string.Join(nl, meta?.Items?.Select(x => $"Timestep : {x.Timestep} TimeOffset: {x.StartTime} Char: {x.Character}"));
            return retval;
        }

        static void Main(string[] args)
        {
            string model = null;
            string alphabet = null;
            string lm = null;
            string trie = null;
            string audio = null;
            bool extended = false;

            //for (int i = 0; i < 500; i+=16)
            //{
            //    Console.WriteLine(i);
            //}


            //for (int i = 0; i <= 500/16;i++)
            //{
            //    int j = i * 16;
            //    Console.WriteLine(j);
            //}


            // var resultFeatures = LpcNetNativeImp.encodepcm("out.pcm", "features.bin");
            // var resultPcm = LpcNetNativeImp.decodepcm("features.bin", "out.f16");
            //var resultPcm1 = LpcNetNativeImp.synthesize_features("compressed.f32", "sintt1.pcm");
            //var resultPcm2 = LpcNetNativeImp.synthesize_features("compressed.f32", "sintt2.pcm");
            LpcNetNativeImp.init("epachuko/inference_model.pbmm");
            LpcNetNativeImp.tts("soy una voz artificial creada por carlos, me llamo yarvis y dominare el mundo.~");
            // LpcNetNativeImp.tts("soy una voz artificial creada por carlos, me llamo yarvis y dominare el mundo.~");
            File.Delete("output.pcm");
            Stopwatch sw = new Stopwatch();
            sw.Start();
            
            LpcNetNativeImp.tts("soy una voz artificial creada por carlos, me llamo yarvis y dominare el mundo.~");
            //LpcNetNativeImp.synthesize_features("inference_model_cpp.f32", "tt.pcm", 1);
            Console.WriteLine("==============      COMPLETED       ===================");
            
            ConvertToWav(new FileInfo("output.pcm"));
            //ConvertToWav(new FileInfo("tt.pcm"));
            //var resultPcm3 = LpcNetNativeImp.synthesize_features("let.f32", "let.pcm",1);
            //LpcNetNativeImp.synthesize_features("f32_for_lpcnet.f32", "lelo.pcm", 1);
            Console.WriteLine($"{sw.Elapsed}");
            var rocessInfo = new ProcessStartInfo("output.wav");
            Process process = new Process() { StartInfo = rocessInfo };
            process.Start();
           // WaveReader mp3Reader = new WavFileReader("example.mp3");
            return;
            Console.ReadLine();

            LpcNetNativeImp.synthesize_features("tuxlpc4.f32", "tuxlpc4.pcm", 1);
            LpcNetNativeImp.synthesize_features("tuxlpc5.f32", "tuxlpc5.pcm", 1);
            Console.ReadLine();

            //var lpcnet = LpcNetNativeImp.lpcnet_create();
            //var created = LpcNetNativeImp.lpcnet_init(lpcnet);
            //var size = LpcNetNativeImp.lpcnet_get_size();
            //var encoder = LpcNetNativeImp.lpcnet_encoder_create();
            //var encoderInit = LpcNetNativeImp.lpcnet_encoder_init(encoder);
            if (args.Length > 0)
            {
                model = GetArgument(args, "--model");
                alphabet = GetArgument(args, "--alphabet");
                lm = GetArgument(args, "--lm");
                trie = GetArgument(args, "--trie");
                audio = GetArgument(args, "--audio");
                extended = !string.IsNullOrWhiteSpace(GetArgument(args, "--extended"));
            }

            const uint N_CEP = 26;
            const uint N_CONTEXT = 9;
            const uint BEAM_WIDTH = 500;
            const float LM_ALPHA = 0.75f;
            const float LM_BETA = 1.85f;

            Stopwatch stopwatch = new Stopwatch();

            using (IDeepSpeech sttClient = new DeepSpeech())
            {
                try
                {
                    Console.WriteLine("Loading model...");
                    stopwatch.Start();
                    sttClient.CreateModel(
                        model ?? "output_graph.pbmm",
                        N_CEP, N_CONTEXT,
                        alphabet ?? "alphabet.txt",
                        BEAM_WIDTH);
                    stopwatch.Stop();

                    Console.WriteLine($"Model loaded - {stopwatch.Elapsed.Milliseconds} ms");
                    stopwatch.Reset();
                    if (lm != null)
                    {
                        Console.WriteLine("Loadin LM...");
                        sttClient.EnableDecoderWithLM(
                            alphabet ?? "alphabet.txt",
                            lm ?? "lm.binary",
                            trie ?? "trie",
                            LM_ALPHA, LM_BETA);

                    }

                    string audioFile = audio ?? "arctic_a0024.wav";
                    var waveBuffer = new WaveBuffer(File.ReadAllBytes(audioFile));
                    using (var waveInfo = new WaveFileReader(audioFile))
                    {
                        Console.WriteLine("Running inference....");

                        stopwatch.Start();

                        string speechResult;
                        if (extended)
                        {
                            Metadata metaResult = sttClient.SpeechToTextWithMetadata(waveBuffer.ShortBuffer, Convert.ToUInt32(waveBuffer.MaxSize / 2), 16000);
                            speechResult = MetadataToString(metaResult);
                        }
                        else
                        {
                            speechResult = sttClient.SpeechToText(waveBuffer.ShortBuffer, Convert.ToUInt32(waveBuffer.MaxSize / 2), 16000);
                        }

                        stopwatch.Stop();

                        Console.WriteLine($"Audio duration: {waveInfo.TotalTime.ToString()}");
                        Console.WriteLine($"Inference took: {stopwatch.Elapsed.ToString()}");
                        Console.WriteLine((extended ? $"Extended result: " : "Recognized text: ") + speechResult);
                    }
                    waveBuffer.Clear();
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex.Message);
                }
            }
        }

        private static void ConvertToWav(FileInfo file)
        {
            using (var stream = file.Open(FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                var waveFormat = new WaveFormat(16000, 16, 1);
                var rawSource = new RawSourceWaveStream(stream, waveFormat);
                using (var fileWriter = new WaveFileWriter(file.Name.Replace(".pcm", ".wav"), waveFormat))
                {
                    rawSource.CopyTo(fileWriter);
                }

            }
        }
    }
}