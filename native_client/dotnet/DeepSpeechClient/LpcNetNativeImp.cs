using System;
using System.Runtime.InteropServices;

namespace DeepSpeechClient
{
    public unsafe struct LPCNetDecState
    {
        LPCNetState lpcnet_state; 
    };

    public unsafe struct LPCNetEncState
    {
        float mem_preemph;
        int pcount;
        float pitch_filt;
        float pitch_max_path_all;
        int best_i;
        float last_gain;
        int last_period;
        int exc_mem;
    };

    public unsafe struct LPCNetState
    {
        int last_exc;
        int frame_count;
        float deemph_mem;
    };
    public static class LpcNetNativeImp
    {
        #region Native Implementation

        [DllImport("lpcnet.so", CallingConvention = CallingConvention.Cdecl, EntryPoint = "lpcnet_create")]
        public static extern unsafe LPCNetState* lpcnet_create();

        [DllImport("lpcnet.so", CallingConvention = CallingConvention.Cdecl, EntryPoint = "lpcnet_get_size")]
        public static extern int lpcnet_get_size();

        [DllImport("lpcnet.so", CallingConvention = CallingConvention.Cdecl, EntryPoint = "lpcnet_decoder_get_size")]
        public static extern int lpcnet_decoder_get_size();

        [DllImport("lpcnet.so", CallingConvention = CallingConvention.Cdecl, EntryPoint = "lpcnet_decoder_init")]
        public static extern unsafe int lpcnet_decoder_init(LPCNetDecState* st);

        [DllImport("lpcnet.so", CallingConvention = CallingConvention.Cdecl, EntryPoint = "lpcnet_decoder_destroy")]
        public static extern unsafe void lpcnet_decoder_destroy(LPCNetDecState* st);

        [DllImport("lpcnet.so", CallingConvention = CallingConvention.Cdecl, EntryPoint = "lpcnet_decode")]
        public static extern unsafe int lpcnet_decode(LPCNetDecState* st,   string buf, short[] pcm);

        [DllImport("lpcnet.so", CallingConvention = CallingConvention.Cdecl, EntryPoint = "lpcnet_encoder_get_size")]
        public static extern unsafe int lpcnet_encoder_get_size();

        [DllImport("lpcnet.so", CallingConvention = CallingConvention.Cdecl, EntryPoint = "lpcnet_encoder_init")]
        public static extern unsafe int lpcnet_encoder_init(LPCNetEncState* st);

        [DllImport("lpcnet.so", CallingConvention = CallingConvention.Cdecl, EntryPoint = "lpcnet_encoder_create")]
        public static extern unsafe LPCNetEncState* lpcnet_encoder_create();
       
        [DllImport("lpcnet.so", CallingConvention = CallingConvention.Cdecl, EntryPoint = "lpcnet_decoder_destroy")]
        public static extern unsafe void lpcnet_encoder_destroy(LPCNetEncState* st);

        [DllImport("lpcnet.so", CallingConvention = CallingConvention.Cdecl, EntryPoint = "lpcnet_encode")]
        public static extern unsafe int lpcnet_encode(LPCNetEncState* st,   short[] pcm,   string buf);

        [DllImport("lpcnet.so", CallingConvention = CallingConvention.Cdecl, EntryPoint = "lpcnet_compute_features")]
        public static extern unsafe int lpcnet_compute_features(LPCNetEncState* st,  short[] pcm, float[][] features);

        [DllImport("lpcnet.so", CallingConvention = CallingConvention.Cdecl, EntryPoint = "lpcnet_init")]
        public static extern unsafe int lpcnet_init(LPCNetState* st);

        [DllImport("lpcnet.so", CallingConvention = CallingConvention.Cdecl, EntryPoint = "lpcnet_destroy")]
        public static extern unsafe void lpcnet_destroy(LPCNetState* st);

        [DllImport("lpcnet.so", CallingConvention = CallingConvention.Cdecl, EntryPoint = "lpcnet_synthesize")]
        public static extern unsafe void lpcnet_synthesize(LPCNetState* st,   float[] features, short[] output, int N);

        [DllImport(@"libdeepspeech.so", CallingConvention = CallingConvention.Cdecl, EntryPoint = "decodepcm")]
        public static extern unsafe int decodepcm(string input, string output);

        [DllImport(@"libdeepspeech.so", CallingConvention = CallingConvention.Cdecl, EntryPoint = "encodepcm")]
        public static extern unsafe int encodepcm(string input, string output);

        [DllImport(@"C:/users/neoxz/_bazel_neoxz/yiuexqz6/execroot/org_tensorflow/bazel-out/x64_windows-opt/bin/native_client/libdeepspeech.so", CallingConvention = CallingConvention.Cdecl, EntryPoint = "synthesize_features")]
        public static extern unsafe int synthesize_features(string input, string output, int use_taco);

        [DllImport(@"C:/users/neoxz/_bazel_neoxz/yiuexqz6/execroot/org_tensorflow/bazel-out/x64_windows-opt/bin/native_client/libdeepspeech.so", CallingConvention = CallingConvention.Cdecl, EntryPoint = "tts")]
        public static extern unsafe void tts(string text);
        [DllImport(@"C:/users/neoxz/_bazel_neoxz/yiuexqz6/execroot/org_tensorflow/bazel-out/x64_windows-opt/bin/native_client/libdeepspeech.so", CallingConvention = CallingConvention.Cdecl, EntryPoint = "init")]
        public static extern unsafe void init(string modelpath);
        #endregion
    }
}