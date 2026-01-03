import Link from "next/link"
import { BookOpen, TrendingUp, Shield, Code, Users, ChevronRight } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"

const subjects = [
  {
    id: "ml-cyber",
    title: "Machine Learning for Cyber Security",
    description: "Learn how ML algorithms detect threats, classify malware, and predict security vulnerabilities.",
    icon: TrendingUp,
    color: "text-blue-500",
    units: 3,
    progress: 65,
  },
  {
    id: "firewall-utm",
    title: "Firewall and UTM Architecture",
    description: "Master firewall configurations, UTM systems, and network security architecture principles.",
    icon: Shield,
    color: "text-green-500",
    units: 3,
    progress: 45,
  },
  {
    id: "ethical-hacking",
    title: "Ethical Hacking and Network Defense",
    description: "Explore penetration testing, vulnerability assessment, and defense strategies.",
    icon: Code,
    color: "text-red-500",
    units: 3,
    progress: 80,
  },
  {
    id: "oomd",
    title: "Object Oriented Modelling and Design",
    description: "Understand UML diagrams, design patterns, and object-oriented software engineering.",
    icon: BookOpen,
    color: "text-purple-500",
    units: 3,
    progress: 55,
  },
  {
    id: "life-skills",
    title: "Life Skills for Engineers",
    description: "Develop communication, teamwork, leadership, and professional skills for career success.",
    icon: Users,
    color: "text-orange-500",
    units: 3,
    progress: 90,
  },
]

export default function HomePage() {
  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
      <section className="border-b border-border bg-gradient-to-b from-background to-muted/20">
        <div className="container mx-auto px-4 py-16 md:py-24">
          <div className="max-w-3xl">
            <h1 className="text-4xl md:text-6xl font-bold text-balance mb-6">B.Tech Cyber Security Study Hub</h1>
            <p className="text-xl text-muted-foreground leading-relaxed mb-8">
              Your complete semester-wise learning companion. Master cybersecurity concepts with structured units,
              detailed explanations, diagrams, and exam-focused practice problems.
            </p>
            <div className="flex flex-wrap gap-4">
              <Button asChild size="lg">
                <Link href="/subjects">
                  Browse Subjects <ChevronRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
              <Button asChild variant="outline" size="lg">
                <Link href="/subjects/ml-cyber">Start Learning</Link>
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="container mx-auto px-4 py-12">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-6 text-center">
              <div className="text-3xl font-bold text-primary mb-2">5</div>
              <div className="text-sm text-muted-foreground">Core Subjects</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-6 text-center">
              <div className="text-3xl font-bold text-primary mb-2">15</div>
              <div className="text-sm text-muted-foreground">Units Total</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-6 text-center">
              <div className="text-3xl font-bold text-primary mb-2">45+</div>
              <div className="text-sm text-muted-foreground">Topics Covered</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-6 text-center">
              <div className="text-3xl font-bold text-primary mb-2">100+</div>
              <div className="text-sm text-muted-foreground">Practice Problems</div>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Subjects Grid */}
      <section className="container mx-auto px-4 py-12">
        <div className="mb-8">
          <h2 className="text-3xl font-bold mb-2">Semester Subjects</h2>
          <p className="text-muted-foreground">Click any subject to start learning unit by unit</p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {subjects.map((subject) => {
            const Icon = subject.icon
            return (
              <Link key={subject.id} href={`/subjects/${subject.id}`}>
                <Card className="h-full transition-all hover:shadow-lg hover:border-primary/50">
                  <CardHeader>
                    <div className="flex items-start justify-between mb-2">
                      <div className={`p-3 rounded-lg bg-muted ${subject.color}`}>
                        <Icon className="h-6 w-6" />
                      </div>
                      <span className="text-sm text-muted-foreground">{subject.units} Units</span>
                    </div>
                    <CardTitle className="text-xl">{subject.title}</CardTitle>
                    <CardDescription className="leading-relaxed">{subject.description}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">Progress</span>
                        <span className="font-medium">{subject.progress}%</span>
                      </div>
                      <Progress value={subject.progress} className="h-2" />
                    </div>
                  </CardContent>
                </Card>
              </Link>
            )
          })}
        </div>
      </section>

      {/* Features Section */}
      <section className="container mx-auto px-4 py-16 border-t border-border">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-12">Everything You Need to Excel</h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="w-12 h-12 rounded-full bg-primary/10 text-primary flex items-center justify-center mx-auto mb-4">
                <BookOpen className="h-6 w-6" />
              </div>
              <h3 className="font-semibold mb-2">Structured Learning</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Unit-wise breakdown with clear progression through topics
              </p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 rounded-full bg-primary/10 text-primary flex items-center justify-center mx-auto mb-4">
                <Code className="h-6 w-6" />
              </div>
              <h3 className="font-semibold mb-2">Practical Examples</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Real-world problems with step-by-step solutions
              </p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 rounded-full bg-primary/10 text-primary flex items-center justify-center mx-auto mb-4">
                <Shield className="h-6 w-6" />
              </div>
              <h3 className="font-semibold mb-2">Exam Ready</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Practice questions and exam tips for every topic
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
