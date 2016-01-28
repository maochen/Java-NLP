package org.maochen.nlp.parser.stanford.pcfg;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.maochen.nlp.parser.DTree;
import org.maochen.nlp.parser.stanford.StanfordParser;
import org.maochen.nlp.parser.stanford.util.StanfordConst;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.parser.common.ParserQuery;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.semgraph.SemanticGraphFactory;
import edu.stanford.nlp.trees.EnglishGrammaticalStructure;
import edu.stanford.nlp.trees.GrammaticalStructure;
import edu.stanford.nlp.trees.SemanticHeadFinder;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.ArrayCoreMap;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.ScoredObject;

/**
 * Created by Maochen on 12/8/14.
 */
public class StanfordPCFGParser extends StanfordParser {

    private static final Logger LOG = LoggerFactory.getLogger(StanfordPCFGParser.class);

    private LexicalizedParser parser = null;

    // This is for the backward compatibility
    public void tagPOS(List<CoreLabel> tokens, Tree tree) {
        try {
            List<TaggedWord> posList = tree.getChild(0).taggedYield();
            for (int i = 0; i < tokens.size(); i++) {
                String pos = posList.get(i).tag();
                tokens.get(i).setTag(pos);
            }
        } catch (Exception e) {
            tagPOS(tokens); // At least gives you something.
            LOG.warn("POS Failed:\n" + tree.pennString());
        }
    }

    // This is for coref using.
    public Pair<CoreMap, GrammaticalStructure> parseForCoref(String sentence) {
        List<CoreLabel> tokens = stanfordTokenize(sentence);
        Tree tree = parser.parse(tokens);
        GrammaticalStructure gs = tagDependencies(tree, true);
        tagPOS(tokens);
        tagLemma(tokens);
        tagNamedEntity(tokens);

        CoreMap result = new ArrayCoreMap();
        result.set(CoreAnnotations.TokensAnnotation.class, tokens);
        result.set(TreeCoreAnnotations.TreeAnnotation.class, tree);

        GrammaticalStructure.Extras extras = GrammaticalStructure.Extras.NONE;
        SemanticGraph deps = SemanticGraphFactory.generateCollapsedDependencies(gs, extras);
        SemanticGraph uncollapsedDeps = SemanticGraphFactory.generateUncollapsedDependencies(gs, extras);
        SemanticGraph ccDeps = SemanticGraphFactory.generateCCProcessedDependencies(gs, extras);

        result.set(SemanticGraphCoreAnnotations.CollapsedDependenciesAnnotation.class, deps);
        result.set(SemanticGraphCoreAnnotations.BasicDependenciesAnnotation.class, uncollapsedDeps);
        result.set(SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation.class, ccDeps);
        return new ImmutablePair<>(result, gs);
    }

    /**
     * This is a piece of mystery code. It allows copula as head!!! Dont touch this unless you have
     * full confidence. This code cannot be found in their Javadoc.... By Maochen
     */
    // What is Mary happy about? -- copula
    private GrammaticalStructure tagDependencies(Tree tree, boolean makeCopulaVerbHead) {
        SemanticHeadFinder headFinder = new SemanticHeadFinder(!makeCopulaVerbHead); // keep copula verbs as head
        // string -> true return all tokens including punctuations.
        GrammaticalStructure gs = new EnglishGrammaticalStructure(tree, string -> true, headFinder);
        return gs;
    }

    @Override
    public DTree parse(final String sentence) {
        if (sentence == null || sentence.trim().isEmpty()) {
            return null;
        }
        List<CoreLabel> tokens = stanfordTokenize(sentence);
        // Parse right after get through tokenizer.
        Tree tree = parser.parse(tokens);
        GrammaticalStructure gs = tagDependencies(tree, true);

        tagPOS(tokens, tree);
        tagLemma(tokens);
        tagNamedEntity(tokens);

        DTree dTree = StanfordTreeBuilder.generate(tokens, gs.typedDependencies(), null);
        dTree.setOriginalSentence(sentence);
        return dTree;
    }


    public Table<DTree, Tree, Double> getKBestParse(String sentence, int k) {
        if (parser == null) {
            LOG.info("Use default PCFG model.");
            parser = LexicalizedParser.loadModel();
        }

        List<CoreLabel> tokens = stanfordTokenize(sentence);

        // Parse right after get through tokenizer.
        ParserQuery pq = parser.parserQuery();
        pq.parse(tokens);
        List<ScoredObject<Tree>> scoredTrees = pq.getKBestPCFGParses(k);

        Table<DTree, Tree, Double> result = HashBasedTable.create();
        for (ScoredObject<Tree> scoredTuple : scoredTrees) {
            Tree tree = scoredTuple.object();
            tagPOS(tokens, tree);
            tagLemma(tokens);
            tagNamedEntity(tokens);

            GrammaticalStructure gs = tagDependencies(tree, true);
            DTree depTree = StanfordTreeBuilder.generate(tokens, gs.typedDependencies(), null);
            result.put(depTree, tree, scoredTuple.score());
        }
        return result;
    }

    public StanfordPCFGParser() {
        this(null, null, null);
    }

    public StanfordPCFGParser(String modelPath, String posTaggerModel, List<String> ners) {
        if (modelPath == null || modelPath.trim().isEmpty()) {
            modelPath = StanfordConst.STANFORD_DEFAULT_PCFG_EN_MODEL;
        }

        parser = LexicalizedParser.loadModel(modelPath, new ArrayList<>());
        super.load(posTaggerModel, ners);
    }
}
