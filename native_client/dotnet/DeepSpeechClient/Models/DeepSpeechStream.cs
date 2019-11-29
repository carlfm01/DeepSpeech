using System;

namespace DeepSpeechClient.Models
{
    /// <summary>
    /// Wrapper of the pointer used for the decoding stream.
    /// </summary>
    public class DeepSpeechStream
    {
        public unsafe IntPtr** StreamingStatePP;

        /// <summary>
        /// Gets or sets the status of the stream.
        /// </summary>
        public bool Active { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="DeepSpeechStream"/> class.
        /// </summary>
        public DeepSpeechStream()
        {
            
        }

        /// <summary>
        /// Initializes a new instance.
        /// </summary>
        /// <param name="streamingStatePP">Native pointer.</param>
        public unsafe DeepSpeechStream(IntPtr** streamingStatePP)
        {
            StreamingStatePP = streamingStatePP;
            Active = true;
        }
    }
}
