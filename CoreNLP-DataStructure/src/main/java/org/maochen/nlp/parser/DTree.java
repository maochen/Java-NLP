package org.maochen.nlp.parser;

import org.apache.commons.lang3.StringUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

/**
 * Author: Maochen.G   contact@maochen.org License: check the LICENSE file. <p> This follows CoNLL-X
 * shared task: Multi-Lingual Dependency Parsing Format <p> Created by Maochen on 12/8/14.
 */
public class DTree extends ArrayList<DNode> {
  private DNode padding;

  private static final String UUID_KEY = "uuid";
  private static final String SENTENCE_TYPE_KEY = "sentence_type";

  @Override
  public String toString() {
    return this.stream()
        .map(DNode::toString)
        .collect(Collectors.joining(System.lineSeparator())).trim();
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
    return UUID.fromString(padding.getFeature(UUID_KEY));
  }

  public void setUUID(UUID id) {
    padding.addFeature(UUID_KEY, id.toString());
  }

  public String sentence() {

    if (!this.padding.getChildren().iterator().hasNext()) {
      return StringUtils.EMPTY;
    }

    // No index_start
    if (this.padding.getChildren().iterator().next().getFeature("index_start") == null) {
      throw new RuntimeException("No idx_start, idx_end");
    }

    int lastIndex = 0;
    StringBuilder stringBuilder = new StringBuilder();

    for (int i = 1; i < this.size(); i++) {
      DNode node = this.get(i);
      int idxStart = Integer.parseInt(node.getFeature("index_start"));
      int idxEnd = Integer.parseInt(node.getFeature("index_end"));

      if (lastIndex + 1 == idxStart) {
        stringBuilder.append(StringUtils.SPACE);
      }
      stringBuilder.append(node.getForm());
      lastIndex = idxEnd;
    }
    return stringBuilder.toString();
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
    padding.setcPOSTag(LangLib.CPOSTAG_X);
    padding.setPOS(LangLib.POS_FW);
    padding.setDepLabel(LangLib.DEP_ATTR);
    setUUID(UUID.randomUUID());
  }
}
