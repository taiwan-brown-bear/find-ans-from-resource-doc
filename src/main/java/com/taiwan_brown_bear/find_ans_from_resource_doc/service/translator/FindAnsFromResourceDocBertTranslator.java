package com.taiwan_brown_bear.find_ans_from_resource_doc.service.translator;

import ai.djl.modality.nlp.DefaultVocabulary;
import ai.djl.modality.nlp.bert.BertToken;
import ai.djl.modality.nlp.bert.BertTokenizer;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.qa.QAInput;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import com.taiwan_brown_bear.find_ans_from_resource_doc.configuration.FindAnsFromResourceDocConfiguration;
import lombok.extern.slf4j.Slf4j;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

@Slf4j
public class FindAnsFromResourceDocBertTranslator implements Translator<QAInput, String> {// input, output

    private FindAnsFromResourceDocConfiguration conf;

    private BertTokenizer tokenizer;
    private Vocabulary    vocabulary;
    private List<String> encodedTokenTokens;

    public FindAnsFromResourceDocBertTranslator(FindAnsFromResourceDocConfiguration conf){
        this.conf = conf;
    }

    @Override
    public void prepare(TranslatorContext ctx) throws IOException {
        tokenizer = new BertTokenizer();
        Path path = Paths.get(conf.getLOCAL_VOCABULARY_PYTORCH_MODEL_DIRECTORY_URL() + conf.getLOCAL_VOCABULARY_FILE_NAME());
        vocabulary = DefaultVocabulary.builder()
                .optMinFrequency(1)
                .addFromTextFile(path)
                .optUnknownToken("[UNK]")
                .build();
        // Note: The deserialized vocabulary maintains two mappings
        //       1) word  -> index
        //       2) index -> word
        //       Internally, the index should be converted to word vector.

        if (log.isDebugEnabled()) {
            final String WORD = "car";
            long   wordIndex = vocabulary.getIndex(WORD);
            String wordToken = vocabulary.getToken(wordIndex);
            log.debug("The index of \"{}\" is \"{}\"", WORD, wordIndex);
            log.debug("The token of the index \"{}\" is \"{}\"", wordIndex, wordToken);
            // e.g., The index of "car" is "2482"
            // e.g., The token of the index "2482" is "car"
        }
    }

    @Override
    public NDList processInput(TranslatorContext ctx, QAInput input)
    {
        if (log.isDebugEnabled())
        {
            String providedResourceDoc = "BBC Japan was a general entertainment Channel. Which operated between December 2004 and April 2006. It ceased operations after its Japanese distributor folded.";
            String question = "What is BBC?";
            List<String> tokensOfProvidedResourceDoc = tokenizer.tokenize(providedResourceDoc.toLowerCase());
            List<String> tokensOfQuestion            = tokenizer.tokenize(           question.toLowerCase());
            log.debug("Tokens of Provided Resource Doc: " + tokensOfProvidedResourceDoc);
            log.debug("Tokens of Question             : " + tokensOfQuestion);
            BertToken encodedTokensOfQuestionAndResourceDoc = tokenizer.encode(question.toLowerCase(), providedResourceDoc.toLowerCase());
            log.debug("Encoded Tokens of Question & Doc: " + encodedTokensOfQuestionAndResourceDoc.getTokens());
            log.debug("Encoded Token Types of Q.  & Doc: " + encodedTokensOfQuestionAndResourceDoc.getTokenTypes());
            log.debug("Valid Length of Encoded Tokens  : " + encodedTokensOfQuestionAndResourceDoc.getValidLength());
            // e.g., Tokens of Provided Resource Doc:                                      [bbc, japan, was, a, general, entertainment, channel, ., which, operated, between, december, 2004, and, april, 2006, ., it, ceased, operations, after, its, japanese, distributor, folded, .]
            // e.g., Tokens of Question             :         [what, is,     bbc, ?]
            // e.g.,*Encoded Tokens of Question & Doc: [[CLS], what, is,     bbc, ?, [SEP], bbc, japan, was, a, general, entertainment, channel, ., which, operated, between, december, 2004, and, april, 2006, ., it, ceased, operations, after, its, japanese, distributor, folded, ., [SEP]]
            // e.g., Encoded Token Types of Q.  & Doc: [    0,    0,  0,       0, 0,     0,   1,     1,   1, 1,       1,             1,       1, 1,     1,        1,       1,        1,    1,   1,     1,    1, 1,  1,      1,          1,     1,   1,        1,           1,      1, 1,     1]
            // e.g., Valid Length of Encoded Tokens  : 30 i.e.,   1   2        3  4           5      6    7  8        9             10       11 12     13        14       15        16    17   18     19    20 21  22      23          24     25   26        27           28      30
            // e.g., word's indices                  :    101, 2054, ...
            // e.g., attentionMask                   :      1,    1,  1,       1, 1,     1,   1,     1,   1, 1,       1,             1,       1, 1,     1,        1,       1,        1,    1,   1,     1,    1, 1,  1,      1,          1,     1,   1,        1,           1,      1, 1,     1
            // e.g., tokenType                       :      0,    0,  0,       0, 0,     0,   1,     1,   1, 1, ...
        }

        BertToken encodedToken = tokenizer.encode(input.getQuestion().toLowerCase(), input.getParagraph().toLowerCase());
        // get the encoded tokens that would be used in precessOutput
        encodedTokenTokens = encodedToken.getTokens();

        long[] indices       = encodedTokenTokens.stream().mapToLong(vocabulary::getIndex).toArray();// map the tokens(String) to indices(long)
        long[] attentionMask = encodedToken      .getAttentionMask().stream().mapToLong(i -> i).toArray();
        long[] tokenType     = encodedToken      .getTokenTypes().stream().mapToLong(i -> i).toArray();
        log.info("      indices.length: {}",       indices.length);
        log.info("attentionMask.length: {}", attentionMask.length);
        log.info("    tokenType.length: {}",     tokenType.length);

        NDManager manager = ctx.getNDManager();
        NDArray indicesArray       = manager.create(indices      );// long array to NDArray
        NDArray attentionMaskArray = manager.create(attentionMask);// long array to NDArray
        NDArray tokenTypeArray     = manager.create(tokenType    );// long array to NDArray

        // 3 NDArrays -> 1 NDList (i.e., model input)
        return new NDList(indicesArray, attentionMaskArray, tokenTypeArray);// The order matters
    }

    @Override
    public String processOutput(TranslatorContext ctx, NDList list) {
        NDArray startLogits = list.get(0);
        NDArray endLogits   = list.get(1);
        int startIdx = (int) startLogits.argMax().getLong();
        int endIdx   = (int) endLogits  .argMax().getLong();
        return encodedTokenTokens.subList(startIdx, endIdx + 1).toString();
    }

    @Override
    public Batchifier getBatchifier() {
        return Batchifier.STACK;
    }
}
