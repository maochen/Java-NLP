package org.maochen.nlp.maochen.nlp.parser.stanford.nn;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.parser.nndep.DependencyParser;
import edu.stanford.nlp.trees.GrammaticalStructure;
import org.maochen.nlp.maochen.nlp.datastructure.DTree;
import org.maochen.nlp.maochen.nlp.datastructure.LangTools;
import org.maochen.nlp.maochen.nlp.parser.IParser;
import org.maochen.nlp.maochen.nlp.parser.StanfordParserUtils;
import org.maochen.nlp.maochen.nlp.parser.stanford.StanfordParser;

import java.util.List;

/**
 * Created by Maochen on 4/6/15.
 */
public class StanfordNNDepParser extends StanfordParser {

    public static DependencyParser nndepParser = null;

    // 4. Dependency Label
    private GrammaticalStructure tagDependencies(List<? extends HasWord> taggedWords) {
        GrammaticalStructure gs = nndepParser.predict(taggedWords);
        return gs;
    }

    @Override
    public DTree parse(String sentence) {
        List<CoreLabel> tokenizedSentence = stanfordTokenize(sentence);
        tagPOS(tokenizedSentence);
        tagLemma(tokenizedSentence);
        GrammaticalStructure gs = tagDependencies(tokenizedSentence);
        tagNamedEntity(tokenizedSentence);
        String conllXString = StanfordParserUtils.getCoNLLXString(gs.typedDependencies(), tokenizedSentence);
        DTree depTree = LangTools.getDTreeFromCoNLLXString(conllXString);
        return depTree;
    }

    public StanfordNNDepParser() {
        this(null, null, false);
    }

    public StanfordNNDepParser(final String inputModelPath, final String posTaggerModel, final boolean initNER) {
        String modelPath = inputModelPath == null || inputModelPath.trim().isEmpty() ? DependencyParser.DEFAULT_MODEL : inputModelPath;
        nndepParser = DependencyParser.loadFromModelFile(modelPath);
        super.load(posTaggerModel, initNER);
    }

    public static void main(String[] args) {
        IParser parser = new StanfordNNDepParser(DependencyParser.DEFAULT_MODEL, null, false);
        String text = "I went to the store and buy a car.";
        DTree tree = parser.parse(text);
        System.out.println(parser.parse(text).toString());
    }
}
