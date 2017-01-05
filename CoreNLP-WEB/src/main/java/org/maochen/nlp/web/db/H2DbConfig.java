package org.maochen.nlp.web.db;

import org.h2.Driver;
import org.h2.server.web.WebServlet;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.web.servlet.ServletRegistrationBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.jpa.repository.config.EnableJpaRepositories;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.datasource.DriverManagerDataSource;
import org.springframework.orm.jpa.JpaTransactionManager;
import org.springframework.orm.jpa.LocalContainerEntityManagerFactoryBean;
import org.springframework.orm.jpa.vendor.Database;
import org.springframework.orm.jpa.vendor.HibernateJpaVendorAdapter;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.annotation.EnableTransactionManagement;

import javax.sql.DataSource;

/**
 * Created by mguan on 12/27/16.
 */
@Configuration
@EnableTransactionManagement
@EnableJpaRepositories
public class H2DbConfig {

  @Value("${spring.data.h2.url}")
  public String dbUrl;

  @Value("${spring.data.h2.username}")
  public String userName;

  @Value("${spring.data.h2.password}")
  public String password;

  /**
   * H2 Data source.
   * jdbc:h2:mem:test
   *
   * @return data source.
   */
  @Bean
  public DataSource dataSource() {
    DriverManagerDataSource dataSource = new DriverManagerDataSource(dbUrl, userName, password);
    dataSource.setDriverClassName(Driver.class.getName());

    // no need shutdown, EmbeddedDatabaseFactoryBean will take care of this
    // EmbeddedDatabaseBuilder builder = new EmbeddedDatabaseBuilder();
    // EmbeddedDatabase dataSource = builder.setType(EmbeddedDatabaseType.H2).setName("h2_example").build();
    return dataSource;
  }

  /**
   * Entity manager factory.
   *
   * @return local container entity manager factory bean.
   */
  @Bean
  public LocalContainerEntityManagerFactoryBean entityManagerFactory() {
    HibernateJpaVendorAdapter jpaVendorAdapter = new HibernateJpaVendorAdapter();
    jpaVendorAdapter.setDatabase(Database.H2);
    jpaVendorAdapter.setGenerateDdl(true);

    LocalContainerEntityManagerFactoryBean entityManagerFactory = new LocalContainerEntityManagerFactoryBean();
    entityManagerFactory.setPackagesToScan(this.getClass().getPackage().getName());
    entityManagerFactory.setJpaVendorAdapter(jpaVendorAdapter);
    entityManagerFactory.setDataSource(dataSource());
    return entityManagerFactory;
  }

  @Bean
  public PlatformTransactionManager transactionManager(){
    return new JpaTransactionManager();
  }

  @Bean
  public JdbcTemplate jdbcTemplate() {
    JdbcTemplate template = new JdbcTemplate(dataSource());
    return template;
  }

  /**
   * H2 console.
   *
   * @return /console/*
   */
  @Bean
  public ServletRegistrationBean h2servletRegistration() {
    ServletRegistrationBean servletRegistrationBean = new ServletRegistrationBean(new WebServlet());
    servletRegistrationBean.addUrlMappings("/h2/*");
    return servletRegistrationBean;
  }
}
