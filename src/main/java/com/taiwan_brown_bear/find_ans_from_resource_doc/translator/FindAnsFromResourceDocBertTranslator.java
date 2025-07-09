package com.taiwan_brown_bear.find_ans_from_resource_doc.translator;

import ai.djl.modality.nlp.DefaultVocabulary;
import ai.djl.modality.nlp.bert.BertToken;
import ai.djl.modality.nlp.bert.BertTokenizer;
import ai.djl.modality.nlp.DefaultVocabulary;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.bert.BertToken;
import ai.djl.modality.nlp.bert.BertTokenizer;
import ai.djl.modality.nlp.qa.QAInput;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.List;

@Slf4j
public class FindAnsFromResourceDocBertTranslator implements Translator<QAInput, String> {

    private List<String>  tokens;
    private Vocabulary    vocabulary;
    private BertTokenizer tokenizer;

    @Override
    public void prepare(TranslatorContext ctx) throws IOException {
        Path path = Paths.get("build/pytorch/bertqa/vocab.txt");
        vocabulary = DefaultVocabulary.builder()
                .optMinFrequency(1)
                .addFromTextFile(path)
                .optUnknownToken("[UNK]")
                .build();
        tokenizer = new BertTokenizer();
    }

    @Override
    public NDList processInput(TranslatorContext ctx, QAInput input) {

        String providedResourceDoc = "abc";
        String question = "eft";

        if (log.isInfoEnabled())
        {
            BertTokenizer tokenizer = new BertTokenizer();

            List<String> tokensOfProvidedResourceDoc = tokenizer.tokenize(providedResourceDoc.toLowerCase());
            List<String> tokensOfQuestion            = tokenizer.tokenize(           question.toLowerCase());
            log.info("Tokens of Provided Resource Doc: " + tokensOfProvidedResourceDoc);
            log.info("Tokens of Question             : " + tokensOfQuestion);
            BertToken encodedTokensOfQuestionAndResourceDoc = tokenizer.encode(question.toLowerCase(), providedResourceDoc.toLowerCase());
            log.info("Encoded Tokens of Question & Doc: " + encodedTokensOfQuestionAndResourceDoc.getTokens());
            log.info("Encoded Token Types of Q.  & Doc: " + encodedTokensOfQuestionAndResourceDoc.getTokenTypes());
            log.info("Valid Length of Encoded Tokens  : " + encodedTokensOfQuestionAndResourceDoc.getValidLength());
            // e.g., Tokens of Provided Resource Doc:                                      [bbc, japan, was, a, general, entertainment, channel, ., which, operated, between, december, 2004, and, april, 2006, ., it, ceased, operations, after, its, japanese, distributor, folded, .]
            // e.g., Tokens of Question             :         [where, is, taiwan, ?]
            // e.g., Encoded Tokens of Question & Doc: [[CLS], where, is, taiwan, ?, [SEP], bbc, japan, was, a, general, entertainment, channel, ., which, operated, between, december, 2004, and, april, 2006, ., it, ceased, operations, after, its, japanese, distributor, folded, ., [SEP]]
            // e.g., Encoded Token Types of Q.  & Doc: [    0,     0,  0,      0, 0,     0,   1,     1,   1, 1,       1,             1,       1, 1,     1,        1,       1,        1,    1,   1,     1,    1, 1,  1,      1,          1,     1,   1,        1,           1,      1, 1,     1]
            // e.g., Valid Length of Encoded Tokens  : 30 i.e.,    1   2       3  4           5      6    7  8        9             10       11 12     13        14       15        16    17   18     19    20 21  22      23          24     25   26        27           28      30
            //
        }


//        if(log.isInfoEnabled()) {
//            var LOCAL_VOCABULARY_FILE_PATH = Paths.get(LOCAL_VOCABULARY_FILE_URL);
//            var deserializedVocabularyWithConfig = DefaultVocabulary.builder()
//                    .optMinFrequency(1)
//                    .addFromTextFile(LOCAL_VOCABULARY_FILE_PATH)
//                    .optUnknownToken("[UNK]")
//                    .build();
//            // Note: The deserialized vocabulary maintains two mappings
//            //       1) word  -> index
//            //       2) index -> word
//            //       Internally, the index should be converted to word vector.
//
//            final String WORD = "car";
//            long wordIndex = deserializedVocabularyWithConfig.getIndex(WORD);
//            String wordToken = deserializedVocabularyWithConfig.getToken(wordIndex);
//            log.info("The index of \"{}\" is \"{}\"", WORD, wordIndex);
//            log.info("The token of the index \"{}\" is \"{}\"", wordIndex, wordToken);
//            // e.g., The index of "car" is "2482"
//            // e.g., The token of the index "2482" is "car"
//        }

        BertToken token =
                tokenizer.encode(
                        input.getQuestion().toLowerCase(),
                        input.getParagraph().toLowerCase());
        // get the encoded tokens that would be used in precessOutput
        tokens = token.getTokens();
        NDManager manager = ctx.getNDManager();
        // map the tokens(String) to indices(long)
        long[] indices = tokens.stream().mapToLong(vocabulary::getIndex).toArray();
        long[] attentionMask = token.getAttentionMask().stream().mapToLong(i -> i).toArray();
        long[] tokenType = token.getTokenTypes().stream().mapToLong(i -> i).toArray();
        NDArray indicesArray = manager.create(indices);
        NDArray attentionMaskArray =
                manager.create(attentionMask);
        NDArray tokenTypeArray = manager.create(tokenType);
        // The order matters
        return new NDList(indicesArray, attentionMaskArray, tokenTypeArray);
    }

    @Override
    public String processOutput(TranslatorContext ctx, NDList list) {
        NDArray startLogits = list.get(0);
        NDArray endLogits = list.get(1);
        int startIdx = (int) startLogits.argMax().getLong();
        int endIdx = (int) endLogits.argMax().getLong();
        return tokens.subList(startIdx, endIdx + 1).toString();
    }

    @Override
    public Batchifier getBatchifier() {
        return Batchifier.STACK;
    }
}
