package org.maochen.nlp.app.relationextract;

import org.apache.commons.lang3.StringUtils;
import org.maochen.nlp.app.relationextract.constant.RelEntityProperty;
import org.maochen.nlp.commons.BinRelation;
import org.maochen.nlp.commons.Entity;
import org.maochen.nlp.parser.DNode;
import org.maochen.nlp.parser.LangLib;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Created by Maochen on 12/15/15.
 */
public class RelExtUtils {
    public static Set<DNode> bfs(DNode root) {

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

    public static void annotateEntityNameByNode(Entity<DNode> entity) {
        entity.suggestedName = entity.stream()
                .map(DNode::getForm).reduce((x1, x2) -> x1 + StringUtils.SPACE + x2).orElse(StringUtils.EMPTY);
    }

    public static List<BinRelation> extractRel(final DNode predicateNode) {
        Set<DNode> usedNodes = new HashSet<>();
        List<Entity<DNode>> resolvedEntities = new ArrayList<>();

        List<DNode> children = predicateNode.getChildren().stream()
                .filter(x -> !x.getDepLabel().equals(LangLib.DEP_PUNCT))
                .filter(x -> !x.getDepLabel().startsWith(LangLib.DEP_AUX)) // Remove aux verb.
                .collect(Collectors.toList());

        for (DNode childEntity : children) {
            if (!usedNodes.contains(childEntity)) {
                // TODO: Shallow Parsing might also needs to be here.
                Set<DNode> entityNodes = bfs(childEntity);
                usedNodes.addAll(entityNodes);

                List<DNode> entityNodesList = entityNodes.stream()
                        .sorted((n1, n2) -> Integer.compare(n1.getId(), n2.getId())).collect(Collectors.toList());
                Entity<DNode> entity = new Entity<>();
                entity.addAll(entityNodesList);
                annotateEntityNameByNode(entity);
                resolvedEntities.add(entity);
            }
        }

        List<BinRelation> result = new ArrayList<>();
        for (int i = 0; i < resolvedEntities.size(); i++) {
            Entity<DNode> thisEntity = resolvedEntities.get(i);

            for (int j = i + 1; j < resolvedEntities.size(); j++) {
                Entity<DNode> thatEntity = resolvedEntities.get(j);
                BinRelation binRelation = new BinRelation();
                binRelation.setLeft(thisEntity);
                binRelation.setRight(thatEntity);
                binRelation.setRel(predicateNode.getLemma());
                binRelation.feats.put(RelEntityProperty.REL_ORIGINAL_WORD, predicateNode.getForm());
                binRelation.feats.put(RelEntityProperty.POS, predicateNode.getPOS());
                // TODO: add more to the predicate
                result.add(binRelation);
            }
        }

        return result;
    }

}
