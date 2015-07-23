package org.maochen.nlp.datastructure;

import org.apache.commons.lang3.StringUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * Copyright 2014-2015 maochen.org
 * Author: Maochen.G   contact@maochen.org
 * License: check the LICENSE file.
 * <p>
 * This follows CoNLL-X shared task: Multi-Lingual Dependency Parsing Format
 * <p>
 * Created by Maochen on 12/8/14.
 */
public class DTree extends ArrayList<DNode> {
    private DNode padding;
    private String sentenceType = StringUtils.EMPTY;
    private String originalSentence = StringUtils.EMPTY;

    @Override
    public String toString() {
        return this.stream()
                .filter(x -> x != padding)
                .map(x -> x.toString() + System.lineSeparator())
                .reduce((x, y) -> x + y).get();
    }

    @Override
    public boolean add(DNode node) {
        if (node == null) return false;
        if (this.contains(node)) return false;

        node.setTree(this);
        return super.add(node);
    }

    public List<DNode> getRoots() {
        return padding.getChildren();
    }

    public DNode getPaddingNode() {
        return padding;
    }

    public String getOriginalSentence() {
        return originalSentence;
    }

    public void setOriginalSentence(String originalSentence) {
        this.originalSentence = originalSentence;
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
        padding.setForm("^");
        padding.setLemma("^");
        this.add(padding);
    }
}
