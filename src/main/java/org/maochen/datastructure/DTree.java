package org.maochen.datastructure;

import org.apache.commons.lang3.StringUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by Maochen on 12/8/14.
 */
public class DTree extends ArrayList<DNode> {

    private DNode padding;

    private String sentenceType = StringUtils.EMPTY;

    @Override
    public String toString() {
        return this.stream()
                .filter(x -> x != padding)
                .map(x -> x.toString() + System.lineSeparator())
                .reduce((x, y) -> x + y).get();
    }

    @Override
    public boolean add(DNode node) {
        if (node != null) {
            node.setTree(this);
        }
        return super.add(node);
    }

    public List<DNode> getRoots() {
        return this.stream().parallel().filter(x -> x.getDepLabel().equals(LangLib.DEP_ROOT)).distinct().collect(Collectors.toList());
    }

    public DNode getPaddingNode() {
        return padding;
    }

    public String getSentenceType() {
        return sentenceType;
    }

    public void setSentenceType(String sentenceType) {
        this.sentenceType = sentenceType;
    }

    public DTree() {
        padding = new DNode();
        padding.setId(0);
        padding.setName("^");
        this.add(padding);
    }
}
