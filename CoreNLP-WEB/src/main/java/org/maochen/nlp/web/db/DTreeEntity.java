package org.maochen.nlp.web.db;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.StringUtils;
import org.maochen.nlp.parser.DNode;
import org.maochen.nlp.parser.DTree;
import org.maochen.nlp.parser.LangTools;

import java.util.Date;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;
import javax.persistence.*;
import javax.validation.constraints.NotNull;

/**
 * Created by mguan on 12/30/16.
 */
@Entity
@Table(name = "parse_tree")
public class DTreeEntity {

  @Id
  @Column(name = "parse_tree_uuid")
  public UUID id;

  @Transient
  public DTree expectedParseTree = null;

  @OneToMany
  @OrderBy("idx ASC")
  public List<DNodeEntity> dNodeEntities;

  @Column(unique = true)
  public String sentence;

  @Column
  public String sentenceType;

  @NotNull
  @Column
  public String author;

  @Column
  public Date date;

  @Column
  public String comment = StringUtils.EMPTY;

  private DTreeEntity() {

  }

  public DTreeEntity(DTree expectedParseTree, String author) {
    this.expectedParseTree = expectedParseTree;
    this.author = author;
    this.date = new Date();
    this.id = expectedParseTree.getUUID();
    transformParseTree(expectedParseTree);
  }


  public DTreeEntity(String expectedCoNLLString, String author) {
    this(LangTools.getDTreeFromCoNLLXString(expectedCoNLLString), author);
  }

  @Transient
  public void transformParseTree(DTree tree) {
    this.id = tree.getUUID();
    this.sentence = tree.sentence();
    this.sentenceType = tree.getSentenceType();
    this.dNodeEntities = tree.stream().map(node -> new DNodeEntity()).collect(Collectors.toList());
    for (int i = 1; i < tree.size(); i++) {
      dNodeEntities.get(i).transformDNode(tree.get(i), this.id, dNodeEntities);
    }
    dNodeEntities.remove(0);
  }

  @Transient
  public DTree getDTree() {
    DTree tree = new DTree();
    tree.setUUID(this.id);
    tree.setSentenceType(this.sentenceType);
    List<DNode> nodes = this.dNodeEntities.stream().map(DNodeEntity::getDNode).collect(Collectors.toList());
    tree.addAll(nodes);

    // HEAD
    // Children
    // ....  a lot
    throw new NotImplementedException("WIP");

  }

  @Override
  public String toString() {
    StringBuilder stringBuilder = new StringBuilder();
    stringBuilder.append(this.id.toString()).append("\t");
    stringBuilder.append(this.sentence).append("\t");
    stringBuilder.append(this.expectedParseTree).append("\t");
    stringBuilder.append(this.author).append("\t");
    stringBuilder.append(this.date).append("\t");
    stringBuilder.append(this.comment);
    return stringBuilder.toString();
  }
}
