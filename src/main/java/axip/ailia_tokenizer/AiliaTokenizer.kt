package axip.ailia_tokenizer

import android.util.Log

/**
 * Tokenizer for natural language processing models.
 *
 * This class provides text tokenization (encoding to token IDs) and
 * vocabulary lookup via the ailia Tokenizer native library.
 *
 * @constructor Creates a tokenizer instance of the given type.
 * @param tokenizerType Type of the tokenizer, one of the AILIA_TOKENIZER_TYPE_* constants.
 */
class AiliaTokenizer(
    tokenizerType: Int = AILIA_TOKENIZER_TYPE_XLM_ROBERTA,
) {
    /**
     * Constants of the tokenizer.
     *
     * AILIA_TOKENIZER_TYPE_* defines the tokenizer type passed to the constructor,
     * and AILIA_TOKENIZER_FLAG_* defines the tokenizer option.
     */
    companion object {
        /** Default flag (no option). */
        const val AILIA_TOKENIZER_FLAG_NONE = 0

        /** Tokenizer for Whisper models. */
        const val AILIA_TOKENIZER_TYPE_WHISPER = 0;
        /** Tokenizer for CLIP models. */
        const val AILIA_TOKENIZER_TYPE_CLIP = 1;
        /** Tokenizer for XLM-RoBERTa models. Requires a SentencePiece model file. */
        const val AILIA_TOKENIZER_TYPE_XLM_ROBERTA = 2;
        /** Tokenizer for Marian machine translation models. Requires a SentencePiece model file. */
        const val AILIA_TOKENIZER_TYPE_MARIAN = 3;
        /** WordPiece tokenizer for Japanese BERT models. Requires a MeCab dictionary and a vocabulary file. */
        const val AILIA_TOKENIZER_TYPE_BERT_JAPANESE_WORDPIECE = 4;
        /** Character tokenizer for Japanese BERT models. Requires a MeCab dictionary and a vocabulary file. */
        const val AILIA_TOKENIZER_TYPE_BERT_JAPANESE_CHARACTER = 5;
        /** Tokenizer for T5 models. Requires a SentencePiece model file. */
        const val AILIA_TOKENIZER_TYPE_T5 = 6;
        /** Tokenizer for RoBERTa models. Requires a vocabulary file and a merge file. */
        const val AILIA_TOKENIZER_TYPE_ROBERTA = 7;
        /** Tokenizer for BERT models. Requires a vocabulary file. */
        const val AILIA_TOKENIZER_TYPE_BERT = 8;
        /** Tokenizer for GPT-2 models. Requires a vocabulary file and a merge file. */
        const val AILIA_TOKENIZER_TYPE_GPT2 = 9;
        /** Tokenizer for Llama models. Requires a SentencePiece model file. */
        const val AILIA_TOKENIZER_TYPE_LLAMA = 10;
        /** Tokenizer for Gemma models. Requires a SentencePiece model file. */
        const val AILIA_TOKENIZER_TYPE_GEMMA = 11;
        /** Tokenizer for Qwen2 models with NFC normalization. Requires a vocabulary file and a merge file. */
        const val AILIA_TOKENIZER_TYPE_QWEN2 = 12;

        init {
            System.loadLibrary("ailia_tokenizer")
        }
    }

    private val tag = AiliaTokenizer::class.simpleName
    private var tokenizer: Long = 0

    init {
        tokenizer = create(tokenizerType, AILIA_TOKENIZER_FLAG_NONE)
    }

    /**
     * Loads the resource files required by the tokenizer.
     *
     * The required files depend on the tokenizer type. See the description of
     * each AILIA_TOKENIZER_TYPE_* constant.
     *
     * @param modelPath Path to the SentencePiece model file (spm file), or null if not required.
     * @param vocabPath Path to the vocabulary file, or null if not required.
     * @param mergePath Path to the merge file, or null if not required.
     * @param dictionaryPath Path to the MeCab dictionary directory, or null if not required.
     * @param addedTokensPath Path to added_tokens.json (explicit IDs).
     * @param tokenizerConfigPath Path to tokenizer_config.json or tokenizer.json (special flags).
     */
    fun loadFiles(modelPath: String? = null, vocabPath: String? = null, mergePath: String? = null,
                  dictionaryPath: String? = null, addedTokensPath: String? = null,
                  tokenizerConfigPath: String? = null) {
        modelPath?.let {
            val result = openModelFile(tokenizer, modelPath)
            if (result != 0) {
                Log.d(tag, "Failed to load model file: $result")
            }
        }

        vocabPath?.let {
            val result = openVocabFile(tokenizer, vocabPath)
            if (result != 0) {
                Log.d(tag,"Failed to load vocab file: $result")
            }
        }

        mergePath?.let {
            val result = openMergeFile(tokenizer, mergePath)
            if (result != 0) {
                Log.d(tag,"Failed to load merge file: $result")
            }
        }

        dictionaryPath?.let {
            val result = openDictionaryFile(tokenizer, dictionaryPath)
            if (result != 0) {
                Log.d(tag,"Failed to load dictionary file: $result")
            }
        }
        addedTokensPath?.let {
            val result = openAddedTokensFile(it)
            if (result != 0) Log.d(tag, "Failed to load added tokens: $result")
        }
        tokenizerConfigPath?.let {
            val result = openTokenizerConfigFile(it)
            if (result != 0) Log.d(tag, "Failed to load tokenizer config: $result")
        }

    }

    /** Loads added token IDs after the vocabulary has been loaded. */
    fun openAddedTokensFile(path: String): Int = openAddedTokensFile(tokenizer, path)

    /** Loads added-token attributes from tokenizer_config.json or tokenizer.json. */
    fun openTokenizerConfigFile(path: String): Int = openTokenizerConfigFile(tokenizer, path)

    /** Registers special tokens. Existing IDs are preserved; new IDs follow registration order. */
    fun addSpecialTokens(tokens: Array<String>): Int = addSpecialTokens(tokenizer, tokens)

    /**
     * Encodes a text into token IDs. Special tokens are encoded as plain text,
     * the same as previous releases (equivalent to ailiaTokenizerEncode).
     *
     * @param text Text to tokenize (UTF-8).
     * @return Array of token IDs.
     */
    fun encode(text: String): IntArray = encode(text, splitSpecialTokens = true)

    /** Encodes text keeping registered special tokens (the Hugging Face default). */
    fun encodeWithSpecialTokens(text: String): IntArray = encode(text, splitSpecialTokens = false)

    /** Encodes text, optionally splitting special tokens into ordinary BPE pieces. */
    fun encode(text: String, splitSpecialTokens: Boolean): IntArray {
        val encodeResult = if (splitSpecialTokens) encode(tokenizer, text) else encodeWithSpecialTokens(tokenizer, text)
        if (encodeResult != 0) {
            Log.d(tag,"Failed to encode text: $encodeResult")
        }

        // Get token count
        val tokenCount = getTokenCount(tokenizer)

        // Get tokens
        val tokens = IntArray(tokenCount)
        val tokenResult = getTokens(tokenizer, tokens, tokenCount)
        if (tokenResult != 0) {
            Log.d(tag,"Failed to get tokens: $tokenResult")
        }

        return tokens
    }

    /**
     * Decodes token IDs into text, skipping special tokens
     * (equivalent to ailiaTokenizerDecode).
     *
     * @param tokens Token IDs to decode.
     * @return Decoded text.
     */
    fun decode(tokens: IntArray): String {
        val text = decode(tokenizer, tokens)
        if (text == null) {
            Log.d(tag, "Failed to decode tokens")
            return ""
        }
        return text
    }

    /**
     * Decodes token IDs into text, keeping special tokens
     * (equivalent to ailiaTokenizerDecodeWithSpecialTokens).
     *
     * @param tokens Token IDs to decode.
     * @return Decoded text.
     */
    fun decodeWithSpecialTokens(tokens: IntArray): String {
        val text = decodeWithSpecialTokens(tokenizer, tokens)
        if (text == null) {
            Log.d(tag, "Failed to decode tokens")
            return ""
        }
        return text
    }

    /**
     * Returns the vocabulary word corresponding to a token ID.
     *
     * @param token Token ID.
     * @return Word associated with the token (UTF-8).
     */
    fun getWord(token: Int): String {
        return getVocab(tokenizer, token)
    }

    /**
     * Releases the native resources held by the tokenizer.
     *
     * The instance must not be used after calling this method.
     */
    fun close() {
        destroy(tokenizer)
    }

    private external fun create(tokenizerType: Int, flags: Int): Long
    private external fun destroy(tokenizer: Long)
    private external fun openModelFile(tokenizer: Long, path: String): Int
    private external fun openVocabFile(tokenizer: Long, path: String): Int
    private external fun openMergeFile(tokenizer: Long, path: String): Int
    private external fun openDictionaryFile(tokenizer: Long, path: String): Int
    private external fun openAddedTokensFile(tokenizer: Long, path: String): Int
    private external fun openTokenizerConfigFile(tokenizer: Long, path: String): Int
    private external fun addSpecialTokens(tokenizer: Long, tokens: Array<String>): Int
    private external fun encodeWithSpecialTokens(tokenizer: Long, text: String): Int
    private external fun encode(tokenizer: Long, text: String): Int
    private external fun decode(tokenizer: Long, tokens: IntArray): String?
    private external fun decodeWithSpecialTokens(tokenizer: Long, tokens: IntArray): String?
    private external fun getTokenCount(tokenizer: Long): Int
    private external fun getTokens(tokenizer: Long, tokens: IntArray, count: Int): Int
    private external fun getVocab(tokenizer: Long, token: Int): String
}