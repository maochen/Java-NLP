package org.maochen.nlp.web.db;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.maochen.nlp.parser.DTree;
import org.maochen.nlp.parser.IParser;
import org.maochen.nlp.parser.stanford.pcfg.StanfordPCFGParser;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;

/**
 * Created by mguan on 1/4/17.
 */

@RunWith(SpringJUnit4ClassRunner.class)
@SpringBootTest(classes = H2DbConfig.class)
public class ParseTreeDbTest {

  @Autowired
  private DTreeEntityRepository dTreeEntityRepository;

  @Autowired
  private DNodeEntityRepository DNodeEntityRepository;

  private static final IParser PARSER = new StanfordPCFGParser(null, null, null);

  @Test
  public void parseTreePersistTest() {
    DTree tree = PARSER.parse("I have a car.");
    DTreeEntity parseTreeEntity = new DTreeEntity(tree, "SAMPLE_AUTHOR");
    DNodeEntityRepository.save(parseTreeEntity.dNodeEntities);
    dTreeEntityRepository.save(parseTreeEntity);
    DTreeEntity actualParseTreeEntity = dTreeEntityRepository.findAll().iterator().next();
    System.out.println("");
  }

}
