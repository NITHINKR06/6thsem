import Link from "next/link"
import { BookOpen, TrendingUp, Shield, Code, Users, ChevronRight } from "lucide-react"
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

const subjects = [
  {
    id: "ml-cyber",
    title: "Machine Learning for Cyber Security",
    description:
      "Apply machine learning algorithms to cybersecurity challenges including threat detection, malware classification, and anomaly detection.",
    icon: TrendingUp,
    color: "text-blue-500",
    bgColor: "bg-blue-500/10",
    units: ["Supervised Learning", "Unsupervised Learning", "Neural Networks"],
  },
  {
    id: "firewall-utm",
    title: "Firewall and UTM Architecture",
    description:
      "Comprehensive study of firewall technologies, unified threat management, and network security architecture design.",
    icon: Shield,
    color: "text-green-500",
    bgColor: "bg-green-500/10",
    units: ["Firewall Fundamentals", "UTM Systems", "Advanced Configurations"],
  },
  {
    id: "ethical-hacking",
    title: "Ethical Hacking and Network Defense",
    description:
      "Master ethical hacking techniques, penetration testing methodologies, and network defense strategies.",
    icon: Code,
    color: "text-red-500",
    bgColor: "bg-red-500/10",
    units: ["Reconnaissance & Scanning", "Exploitation Techniques", "Defense Strategies"],
  },
  {
    id: "oomd",
    title: "Object Oriented Modelling and Design",
    description: "Learn object-oriented principles, UML modeling, design patterns, and software architecture.",
    icon: BookOpen,
    color: "text-purple-500",
    bgColor: "bg-purple-500/10",
    units: ["OOP Concepts", "UML Diagrams", "Design Patterns"],
  },
  {
    id: "life-skills",
    title: "Life Skills for Engineers",
    description:
      "Develop essential soft skills including communication, teamwork, leadership, and professional development.",
    icon: Users,
    color: "text-orange-500",
    bgColor: "bg-orange-500/10",
    units: ["Communication Skills", "Team Collaboration", "Professional Development"],
  },
]

export default function SubjectsPage() {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-12">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-3">All Subjects</h1>
          <p className="text-lg text-muted-foreground">Select a subject to view units and start learning</p>
        </div>

        <div className="space-y-6">
          {subjects.map((subject) => {
            const Icon = subject.icon
            return (
              <Card key={subject.id} className="transition-all hover:shadow-lg">
                <CardHeader>
                  <div className="flex items-start gap-4">
                    <div className={`p-4 rounded-xl ${subject.bgColor} ${subject.color} shrink-0`}>
                      <Icon className="h-8 w-8" />
                    </div>
                    <div className="flex-1">
                      <CardTitle className="text-2xl mb-2">{subject.title}</CardTitle>
                      <CardDescription className="text-base leading-relaxed mb-4">
                        {subject.description}
                      </CardDescription>
                      <div className="flex flex-wrap gap-2 mb-4">
                        {subject.units.map((unit, index) => (
                          <Badge key={index} variant="secondary">
                            Unit {index + 1}: {unit}
                          </Badge>
                        ))}
                      </div>
                      <Button asChild>
                        <Link href={`/subjects/${subject.id}`}>
                          View Subject <ChevronRight className="ml-2 h-4 w-4" />
                        </Link>
                      </Button>
                    </div>
                  </div>
                </CardHeader>
              </Card>
            )
          })}
        </div>
      </div>
    </div>
  )
}
