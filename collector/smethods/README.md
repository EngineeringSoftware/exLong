# smethods

Method level dependency graph.

## Requirements
Java 8

## Usage
```
mvn test-compile
mvn exec:java -Dexec.mainClass=org.smethods.MethodLevelStaticDepsBuilder -Dexec.args="path_to_test_classes"
```

## Background
1. [ASM](http://asm.ow2.org/)
2. [Visitor pattern](https://en.wikipedia.org/wiki/Visitor_pattern)

## Example
```
public class A {
    public static int i = 0;

    public static void inc() {
        i++;
    }

    public static void main(String[] args) {
        B b = new C();
        b.foo();

        inc();
    }
}

class B {
    public void foo() {
        System.out.println("B foo");
    }
}

class C extends B{
    public void foo() {
        System.out.println("C foo");
    }
}
```

```
A#<clinit>() A#i
B#foo() 
A#main([Ljava/lang/String;) A#inc(),B#foo(),C#<init>(),C#foo()
A#inc() A#i
B#<init>() 
A#i A#<clinit>()
C#<init>() B#<init>()
C#foo() 
A#<init>() 
```