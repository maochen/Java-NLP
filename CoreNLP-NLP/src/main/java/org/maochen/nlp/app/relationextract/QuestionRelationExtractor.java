package org.maochen.nlp.app.relationextract;

import com.google.common.collect.ImmutableSet;

import org.maochen.nlp.commons.BinRelation;
import org.maochen.nlp.commons.Entity;
import org.maochen.nlp.parser.DNode;
import org.maochen.nlp.parser.DTree;
import org.maochen.nlp.parser.LangLib;
import org.maochen.nlp.parser.stanford.nn.StanfordNNDepParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Created by Maochen on 11/26/15.
 */
public class QuestionRelationExtractor {

    private static final Logger LOG = LoggerFactory.getLogger(QuestionRelationExtractor.class);

    private static Set<DNode> bfs(DNode root) {

        Set<DNode> result = new HashSet<>();
        Queue<DNode> q = new LinkedList<>();
        q.add(root);

        while (!q.isEmpty()) {
            DNode current = q.poll();
            result.add(current);
            q.addAll(current.getChildren());
        }

        return result;
    }

    private static Entity<DNode> extractRemaining(final DNode wildcardNode, final DNode wildcardHeadVerb, final DTree tree) {
        Set<DNode> remain = wildcardHeadVerb.getChildren().stream().filter(x -> x != wildcardNode).collect(Collectors.toSet());
        Set<DNode> children = remain.stream().map(QuestionRelationExtractor::bfs).flatMap(Collection::stream).collect(Collectors.toSet());
        remain.addAll(children);

        Set<String> excludingLabels = ImmutableSet.of(LangLib.DEP_AUX, LangLib.DEP_AUXPASS);
        remain = remain.stream().filter(x -> !excludingLabels.contains(x.getDepLabel())).collect(Collectors.toSet());
        remain.remove(tree.get(tree.size() - 1));
        Entity<DNode> entity = new Entity<>();
        entity.addAll(remain.stream().sorted((x1, x2) -> Integer.compare(x1.getId(), x2.getId())).collect(Collectors.toList()));
        return entity;
    }

    private static Entity<DNode> getWildcardEntity(DNode wildcardNode, String questionType) {
        Entity<DNode> wildcardEntity = new Entity<>(wildcardNode);
        wildcardEntity.suggestedName = "?X";
        wildcardEntity.feats.put("question_type", questionType);
        return wildcardEntity;
    }

    private static BinRelation extractNonPolar(final DTree tree, final String questionType) {
        if (tree == null) {
            return null;
        }

        DNode wildcardNode = tree.stream().filter(x -> x.getPOS().toLowerCase().startsWith("w")).findFirst().orElse(null);

        if (wildcardNode == null) {
            LOG.warn("Unable to find wildcard node.");
            return null;
        }

        DNode wildcardHeadVerb = wildcardNode;

        while (wildcardHeadVerb != null && !wildcardHeadVerb.getPOS().startsWith("VB")) {
            wildcardHeadVerb = wildcardHeadVerb.getHead();
        }

        if (wildcardHeadVerb == null) {
            LOG.warn("Unable to find verb for wildcard node [" + wildcardNode.getForm() + "]");
            return null;
        }

        BinRelation wildcardRel = new BinRelation();
        wildcardRel.setLeft(getWildcardEntity(wildcardNode, questionType));
        wildcardRel.setRel(wildcardHeadVerb.getLemma());
        wildcardRel.relType = RelType.WILDCARD.toString();
        wildcardRel.setRight(extractRemaining(wildcardNode, wildcardHeadVerb, tree));
        return wildcardRel;
    }

    // POLAR doesnt have wildcard.
    private static BinRelation extractPolar(DTree tree) {
        // TODO: HERE.
        DNode rootVerb = tree.getRoots().get(0);
//        rootVerb.getChildren().
        BinRelation binRelation = new BinRelation();
        return binRelation;
    }

    public static BinRelation extract(DTree tree, String qt) {
        if (qt.equals("YESNO")) {
            return extractPolar(tree);
        } else {
            return extractNonPolar(tree, qt);
        }
    }

    public static void main(String[] args) {
        // What is meant by an expense?
        // (mean [?X] [by an expense])

        StanfordNNDepParser parser = new StanfordNNDepParser();
        DTree question = parser.parse("What is meant by an expense?");
        System.out.println(question);
        BinRelation rel = extract(question, "XXX");

        System.out.println(rel);
    }

}
