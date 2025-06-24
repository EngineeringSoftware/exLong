This is a Maven extension to run Jacoco without changing the project's pom.xml
Steps:

1. mvn package
2. cp target/jacoco-extension-1.0-SNAPSHOT.jar ${MAVEN_HOME}/lib/ext

That's it!

Once you do that you can just `cd` into the project you want to run Jacoco on and run `mvn test` as usual.

The only downside is that you need to `rm ${MAVEN_HOME}/lib/ext/jacoco-extension-1.0-SNAPSHOT.jar` if you don’t want to run Jacoco. What I usually do is to put the extension in a local maven installation and then switch between installations by setting the $MAVEN_HOME variable appropriately.