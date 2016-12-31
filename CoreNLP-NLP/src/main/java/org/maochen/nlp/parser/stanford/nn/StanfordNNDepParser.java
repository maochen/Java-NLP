package org.maochen.nlp.parser.stanford.nn;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.parser.nndep.DependencyParser;
import edu.stanford.nlp.trees.GrammaticalStructure;
import org.maochen.nlp.parser.DTree;
import org.maochen.nlp.parser.stanford.StanfordParser;
import org.maochen.nlp.parser.stanford.util.StanfordParserUtils;

import java.util.List;

/**
 * Created by Maochen on 4/6/15.
 */
public class StanfordNNDepParser extends StanfordParser {

    public DependencyParser nndepParser;

    // 4. Dependency Label
    private GrammaticalStructure tagDependencies(List<? extends HasWord> taggedWords) {
        GrammaticalStructure gs = nndepParser.predict(taggedWords);
        return gs;
    }

    @Override
    public DTree parse(String sentence) {
        if (sentence == null || sentence.trim().isEmpty()) {
            return null;
        }

        List<CoreLabel> tokenizedSentence = stanfordTokenize(sentence);
        tagPOS(tokenizedSentence);
        tagLemma(tokenizedSentence);
        GrammaticalStructure gs = tagDependencies(tokenizedSentence);
        tagNamedEntity(tokenizedSentence);
        DTree dTree = StanfordParserUtils.getDTreeFromCoreNLP(gs.typedDependencies(), tokenizedSentence);
        return dTree;
    }

    public StanfordNNDepParser() {
        this(null, null, null);
    }

    public StanfordNNDepParser(final String inputModelPath, final String posTaggerModel, List<String> nerModelPath) {
        String modelPath = inputModelPath == null || inputModelPath.trim().isEmpty() ? DependencyParser.DEFAULT_MODEL : inputModelPath;
        nndepParser = DependencyParser.loadFromModelFile(modelPath);
        super.load(posTaggerModel, nerModelPath);
    }
}
