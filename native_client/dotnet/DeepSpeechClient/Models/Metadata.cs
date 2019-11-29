﻿namespace DeepSpeechClient.Models
{
    /// <summary>
    /// Stores the entire CTC output as an array of character metadata objects.
    /// </summary>
    public class Metadata
    {
        /// <summary>
        /// Approximated confidence value for this transcription.
        /// </summary>
        public double Confidence { get; set; }
        /// <summary>
        /// List of metada items containing char, timestep, and time offset.
        /// </summary>
        public MetadataItem[] Items { get; set; }
    }
}