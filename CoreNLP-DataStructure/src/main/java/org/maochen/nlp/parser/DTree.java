package org.maochen.nlp.parser;

import org.apache.commons.lang3.StringUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * Author: Maochen.G   contact@maochen.org License: check the LICENSE file. <p> This follows CoNLL-X
 * shared task: Multi-Lingual Dependency Parsing Format <p> Created by Maochen on 12/8/14.
 */
public class DTree extends ArrayList<DNode> {
    private DNode padding;

    private static final String UUID_KEY = "uuid";
    private static final String SENTENCE_TYPE_KEY = "sentence_type";
    private static final String ORIGINAL_SENTENCE_KEY = "original_sentence";

    private String originalSentence = StringUtils.EMPTY;


    @Override
    public String toString() {
        return this.stream()
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

    public UUID getUUID() {
        UUID uuid = UUID.fromString(padding.getFeature(UUID_KEY));
        return uuid;
    }

    public void setUUID(UUID id) {
        padding.addFeature(UUID_KEY, id.toString());
    }

    public String getOriginalSentence() {
        if (!originalSentence.trim().isEmpty()) {
            return originalSentence;
        }

        String originFromFeats = padding.getFeature(ORIGINAL_SENTENCE_KEY);
        return originFromFeats == null ? StringUtils.EMPTY : originFromFeats;
    }

    public void setOriginalSentence(String originalSentence) {
        this.originalSentence = originalSentence;
        padding.addFeature(ORIGINAL_SENTENCE_KEY, originalSentence);
    }

    public String getSentenceType() {
        String sentenceType = padding.getFeature(SENTENCE_TYPE_KEY);
        return sentenceType == null ? StringUtils.EMPTY : sentenceType;
    }

    public void setSentenceType(String sentenceType) {
        padding.addFeature(SENTENCE_TYPE_KEY, sentenceType);
    }

    public DTree() {
        padding = new DNode();
        this.add(padding);

        padding.setId(0);
        padding.setForm("^");
        padding.setLemma("^");
        setUUID(UUID.randomUUID());
    }
}
