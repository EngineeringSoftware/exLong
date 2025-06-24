package edu.illinois.extension;

import java.util.Collections;

import org.apache.maven.AbstractMavenLifecycleParticipant;
import org.apache.maven.MavenExecutionException;
import org.apache.maven.execution.MavenSession;
import org.codehaus.plexus.component.annotations.Component;
import org.apache.maven.project.MavenProject;
import org.apache.maven.model.Dependency;
import org.apache.maven.model.Plugin;
import org.apache.maven.model.PluginExecution;
import org.apache.maven.plugin.MojoFailureException;
import org.codehaus.plexus.util.xml.Xpp3Dom;

// your extension must be a "Plexus" component so mark it with the annotation
@Component( role = AbstractMavenLifecycleParticipant.class, hint = "jacoco")
public class JacocoExtension extends AbstractMavenLifecycleParticipant
{
    @Override
    public void afterProjectsRead( MavenSession session )
        throws MavenExecutionException
    {
        boolean jacoco_found = false;
        boolean surefire_found = false;
        for (MavenProject project : session.getProjects()) {
            for (Plugin plugin : project.getBuild().getPlugins()) {
                if (plugin.getArtifactId().equals("jacoco-maven-plugin") && plugin.getGroupId().equals("org.jacoco")) {
                    jacoco_found = true;
                }
                if (plugin.getArtifactId().equals("maven-surefire-plugin") && plugin.getGroupId().equals("org.apache.maven.plugins")) {
                    surefire_found = true;
                    addArgLine(plugin);
                }
            }
        }
        if (!jacoco_found){
            try {
                configureJacoco(session);
            } catch (MojoFailureException mfe) {
                throw new MavenExecutionException("Unable to install jacoco", mfe);
            }
        }

        if (!surefire_found) {
            try {
                configureSurefire(session);
            } catch (MojoFailureException mfe) {
                throw new MavenExecutionException("Unable to install surefire", mfe);
            }
        }
    }

    private void configureJacoco(MavenSession session) throws MojoFailureException {
       for (MavenProject p : session.getProjects()) {
            Plugin newPlug = new Plugin();
            newPlug.setArtifactId("jacoco-maven-plugin");
            newPlug.setGroupId("org.jacoco");
            newPlug.setVersion("0.8.8");

            PluginExecution exPrepareAgent = new PluginExecution();
            exPrepareAgent.setGoals(Collections.singletonList("prepare-agent"));
            newPlug.addExecution(exPrepareAgent);

            PluginExecution exReport = new PluginExecution();
            exReport.setId("report");
            exReport.setPhase("test");    
            exReport.setGoals(Collections.singletonList("report"));
            newPlug.addExecution(exReport);

            p.getBuild().addPlugin(newPlug);
       }
    }

    private void configureSurefire(MavenSession session) throws MojoFailureException {
        for (MavenProject p : session.getProjects()) {
            Plugin newPlug = new Plugin();
            newPlug.setArtifactId("maven-surefire-plugin");
            newPlug.setGroupId("org.apache.maven.plugins");
            newPlug.setVersion("3.0.0-M8");
            p.getBuild().addPlugin(newPlug);
            addArgLine(newPlug);
        }
    }

    private void addArgLine(Plugin plugin) {
        // =inclbootstrapclasses=true
        String agent = "-javaagent:${settings.localRepository}/org/jacoco/org.jacoco.agent/0.8.8/org.jacoco.agent-0.8.8-runtime.jar";
        Xpp3Dom configuration = (Xpp3Dom) plugin.getConfiguration();
        if (configuration == null) {
            configuration = new Xpp3Dom("configuration");
            plugin.setConfiguration(configuration);
        }
        configuration.setAttribute("argLine", agent);
    }
}
